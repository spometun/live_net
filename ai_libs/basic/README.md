Place for packages and modules
which form well-defined concept which can be reused ***outside of the project***.

Unittest for these concepts are encouraged.

Packages and modules in this folder 
MUST NOT depend on:
1. Anything outside of this folder
2. Heavy third-party libs, including:
pytorch, tensorflow, sagemaker, fiftyone and graphics, e.g. matplotlib

Dependency on common libs like:
numpy, scipy, sklearn, are ok.
Because these libs usually do not cause versions/interoperability issues

Imports of any ai_libs dependencies should 
start from ai_libs, e.g. "import ai_libs.basic.something" or be local, e.g. from ..simple_log import LOG
