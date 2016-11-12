# Introduction

The unit test contains the minimum validation tests for the code. To discover all the unit test cases from Mac OS X or Linux, run

```
bin/list_unit_tests
```

or for Windows, run

```
bin\list_unit_tests.bat
```

As described above, run the following command to launch entire tests from Mac OS X or Linux:

```
bin/unit_tests
```

For Windows, run

```
bin\unit_tests.bat
```

To run specific tests, such as `VertexCenteredVectorGrid3.Fill`, run

```
bin/unit_tests VertexCenteredVectorGrid3.Fill
```

You can also use a pattern to run the tests such as

```
bin/unit_tests VertexCenteredVectorGrid3.*
```

Replace `bin/unit_tests` with `bin\unit_tests.bat` for two commands above when running from Windows.
