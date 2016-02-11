from sympy import symbols, conjugate, sqrt, solve, evaluate

# declare variables
h1, h2, h3, h4 = symbols("h1 h2 h3 h4")

# initial constraint
eq1 = h1 - 2*h2 + 3*h3  # = 0

# wavelet equation 1 and 3
eq2 = h1 + h2 + h3 + h4 -sqrt(2)  # = 0

# wavelet equation 2
eq3 = h1*conjugate(h3) + h2*conjugate(h4)   # = 0

# wavelet equation 4
eq4 = -h1 + h2 - h3 + h4    # = 0

solutions = solve([eq1, eq2, eq3, eq4], [h1, h2, h3, h4])

# verify solutions satisfy equations
var_list = [h1, h2, h3, h4] 
for solution in solutions:
    par = zip(var_list, solution)
    assert eq1.subs(par).evalf(chop=True) == 0 # satisfy eq1
    assert eq2.subs(par).evalf(chop=True) == 0 # satisfy eq2
    assert eq3.subs(par).evalf(chop=True) == 0 # satisfy eq3
    assert eq4.subs(par).evalf(chop=True) == 0 # satisfy eq4
    print par # print solution
