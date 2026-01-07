import json
from dataclasses import dataclass
from typing import Callable, List, Dict, Any


@dataclass
class SingleTestResult:
    question: str
    category: str
    keywords: List[str]
    reference_answer: str
    generated_answer: str
    keyword_score: float
    semantic_overlap: float
    length_ratio: float
    combined_score: float
    passed: bool


@dataclass
class EvalReport:
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_keyword_score: float
    avg_semantic_overlap: float
    avg_length_ratio: float
    avg_combined_score: float
    by_category: Dict[str, Dict[str, Any]]
    tests: List[SingleTestResult]


class RAGEvaluator:
    """
    Simple RAG evaluator for tests.jsonl.
    Each line in tests.jsonl must be a JSON object with:
      - question: str
      - reference_answer: str
      - keywords: list[str]
      - category: str
    """

    def __init__(self, tests_path: str):
        self.tests_path = tests_path
        self.tests = self._load_tests()

    # ---------------------- internal helpers ---------------------- #

    def _load_tests(self) -> List[Dict[str, Any]]:
        tests = []
        with open(self.tests_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tests.append(json.loads(line))
        return tests

    @staticmethod
    def _compute_keyword_score(answer: str, keywords: List[str]) -> float:
        if not keywords:
            return 1.0
        ans = answer.lower()
        hits = sum(1 for k in keywords if k.lower() in ans)
        return hits / len(keywords)

    @staticmethod
    def _compute_semantic_overlap(answer: str, reference: str) -> float:
        ans_words = set(answer.lower().split())
        ref_words = set(reference.lower().split())
        if not ans_words or not ref_words:
            return 0.0
        inter = len(ans_words & ref_words)
        union = len(ans_words | ref_words)
        return inter / union

    @staticmethod
    def _compute_length_ratio(answer: str, reference: str) -> float:
        ans_len = max(len(answer.split()), 1)
        ref_len = max(len(reference.split()), 1)
        ratio = ans_len / ref_len
        # clamp to [0, 1] by penalizing too long answers
        return ratio if ratio <= 1.0 else 1.0 / ratio

    # ------------------------ main API ------------------------ #

    def run_evaluation(
        self,
        answer_fn: Callable[[str], str],
        verbose: bool = False,
        keyword_pass_threshold: float = 0.8,
    ) -> EvalReport:
        results: List[SingleTestResult] = []

        for idx, t in enumerate(self.tests, 1):
            q = t["question"]
            ref = t["reference_answer"]
            keywords = t.get("keywords", [])
            category = t.get("category", "unknown")

            # call your RAG system
            generated = answer_fn(q)

            kw_score = self._compute_keyword_score(generated, keywords)
            sem_overlap = self._compute_semantic_overlap(generated, ref)
            len_ratio = self._compute_length_ratio(generated, ref)

            combined = 0.4 * kw_score + 0.4 * sem_overlap + 0.2 * len_ratio
            passed = kw_score >= keyword_pass_threshold

            result = SingleTestResult(
                question=q,
                category=category,
                keywords=keywords,
                reference_answer=ref,
                generated_answer=generated,
                keyword_score=kw_score,
                semantic_overlap=sem_overlap,
                length_ratio=len_ratio,
                combined_score=combined,
                passed=passed,
            )
            results.append(result)

            if verbose:
                print("\n" + "=" * 80)
                print(f"Test #{idx}")
                print("=" * 80)
                print(f"Question: {q}")
                print(f"Category: {category}")
                print(f"Keywords: {keywords}")
                print(f"Reference: {ref}")
                print("\nGenerated Answer:")
                print(generated)
                print("\nScores:")
                print(f"  Keyword score:       {kw_score:.2f}")
                print(f"  Semantic overlap:    {sem_overlap:.2f}")
                print(f"  Length ratio:        {len_ratio:.2f}")
                print(f"  Combined score:      {combined:.2f}")
                print(f"  Passed (kw >= {keyword_pass_threshold}): {passed}")
                print("=" * 80)

        return self._build_report(results)

    def _build_report(self, results: List[SingleTestResult]) -> EvalReport:
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed
        pass_rate = (passed / total * 100.0) if total else 0.0

        avg_kw = sum(r.keyword_score for r in results) / total if total else 0.0
        avg_sem = sum(r.semantic_overlap for r in results) / total if total else 0.0
        avg_len = sum(r.length_ratio for r in results) / total if total else 0.0
        avg_comb = sum(r.combined_score for r in results) / total if total else 0.0

        by_cat: Dict[str, Dict[str, Any]] = {}
        for r in results:
            c = r.category
            by_cat.setdefault(c, {"total": 0, "passed": 0, "scores": []})
            by_cat[c]["total"] += 1
            if r.passed:
                by_cat[c]["passed"] += 1
            by_cat[c]["scores"].append(r.combined_score)

        for c, info in by_cat.items():
            t = info["total"]
            info["pass_rate"] = info["passed"] / t * 100.0 if t else 0.0
            info["avg_score"] = (
                sum(info["scores"]) / t if t else 0.0
            )
            del info["scores"]

        return EvalReport(
            total=total,
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            avg_keyword_score=avg_kw,
            avg_semantic_overlap=avg_sem,
            avg_length_ratio=avg_len,
            avg_combined_score=avg_comb,
            by_category=by_cat,
            tests=results,
        )

    # ---------------------- printing helpers ---------------------- #

    @staticmethod
    def print_report(report: EvalReport) -> None:
        print("\n" + "=" * 80)
        print("RAG EVALUATION SUMMARY")
        print("=" * 80)
        print(f"Total tests:   {report.total}")
        print(f"Passed:        {report.passed}")
        print(f"Failed:        {report.failed}")
        print(f"Pass rate:     {report.pass_rate:.1f}%")

        print("\nAverage scores (0–1):")
        print(f"  Keyword score:     {report.avg_keyword_score:.2f}")
        print(f"  Semantic overlap:  {report.avg_semantic_overlap:.2f}")
        print(f"  Length ratio:      {report.avg_length_ratio:.2f}")
        print(f"  Combined score:    {report.avg_combined_score:.2f}")

        print("\nBy category:")
        for cat, info in report.by_category.items():
            print(f"  {cat}:")
            print(f"    Tests:      {info['total']}")
            print(f"    Pass rate:  {info['pass_rate']:.1f}%")
            print(f"    Avg score:  {info['avg_score']:.2f}")
        print("=" * 80)

    @staticmethod
    def save_results(report: EvalReport, path: str) -> None:
        data = {
            "summary": {
                "total": report.total,
                "passed": report.passed,
                "failed": report.failed,
                "pass_rate": report.pass_rate,
                "avg_keyword_score": report.avg_keyword_score,
                "avg_semantic_overlap": report.avg_semantic_overlap,
                "avg_length_ratio": report.avg_length_ratio,
                "avg_combined_score": report.avg_combined_score,
                "by_category": report.by_category,
            },
            "tests": [
                {
                    "question": r.question,
                    "category": r.category,
                    "keywords": r.keywords,
                    "reference_answer": r.reference_answer,
                    "generated_answer": r.generated_answer,
                    "keyword_score": r.keyword_score,
                    "semantic_overlap": r.semantic_overlap,
                    "length_ratio": r.length_ratio,
                    "combined_score": r.combined_score,
                    "passed": r.passed,
                }
                for r in report.tests
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
