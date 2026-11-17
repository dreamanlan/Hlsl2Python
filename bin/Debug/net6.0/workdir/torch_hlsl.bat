rem working directory

set workdir=%~dp0
cd %workdir%
cd ..

Hlsl2Python -gen-dsl -gl -torch -vectorize-branch -vectorization %workdir%hlsl_test.hlsl

cd %workdir%

