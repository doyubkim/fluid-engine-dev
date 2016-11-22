# Introduction

The performance test measures the computation timing. To list the entire test cases, run

```
bin/list_perf_tests
```

To run the tests, execute

```
bin/perf_tests
```

Similar to the unit test and manual test, you can pass the test name pattern to run specific tests, such as

```
bin/perf_tests Point*
```

For Windows, use `bin\list_perf_tests.bat` and `bin\perf_tests.bat`.

The measured performance for each test will be printed to the console, such as

```
...
[----------] 1 test from PointHashGridSearcher3
[ RUN      ] PointHashGridSearcher3.Build
[----------] PointHashGridSearcher3::build avg. 0.261923 sec.
[       OK ] PointHashGridSearcher3.Build (2732 ms)
[----------] 1 test from PointHashGridSearcher3 (2733 ms total)
...
```

