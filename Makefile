.PHONY: test test-correctness test-performance clean

test: test-correctness

test-correctness:
	@echo "Running correctness validation..."
	@cd tests && ./run_correctness_tests.sh

test-pytest:
	@echo "Running pytest validation..."
	@pytest tests/test_sparse_pytest.py -v

test-performance:
	@echo "Running performance benchmarks..."
	@python benchmarks/benchmark_sparse.py

clean:
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
	@rm -f tests/*.bin tests/*.txt
