# PyOptimizer
Python implementation of general nonlinear optimizer, this library is simple and just reveals the essential of the non-linear optimizer.

* Project backbone are implemented in backend
* Example use of this repository is implemented in `tests/testMonoBA.py` run
```shell script
python tests/testMonoBA.py
```
  * Generally for a nonlinear problem, we can define the problem with parameters (unknown variables) and constraints (equations).
  * Parameter are abstracted as Vertex, constraints are abstracted as edge. 
  * In a problem, vertices must form edges and these edges must generate a residual. and this code will adjust the values of vertices to get minimum residual.
    * A basic optimization problem of finding `a,b,c` to get minimum `y` in  `y = a * x * x * x + b * x * x + c * x + noise` given a set of `y` and `x` is implemented in `tests.py`. Feel free to adjust codes there to solve other simple problem.
    
- [ ]  [BAL dataset][1] test code might be implemented later but it is simple and nearly implemented in tests/testBA.py.


[1]: http://grail.cs.washington.edu/projects/bal/