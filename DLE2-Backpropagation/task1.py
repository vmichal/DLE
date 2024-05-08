import torch
import numpy as np
import code

w = torch.tensor(1)
x = torch.tensor(2.0)
t = torch.tensor(np.float32(3))
b = torch.tensor(4, dtype = torch.float32)
print(f'{w.dtype = }\n{x.dtype = }\n{t.dtype = }\n{b.dtype = }')

# Ensure all tensors are of type float32
w = torch.tensor(1, dtype=torch.float, requires_grad=True)
x = torch.tensor(2.0)
t = torch.tensor(np.float32(3))
b = torch.tensor(4, dtype = torch.float32)

# 1)
#w.requires_grad = True
a = x + b
y = max(a*w, 0)
l = (y - t)**2 + w**2
# Calling .backward() or torch.autograd.grad() destroys the tree and hence can be done only once.

#print(f'{torch.autograd.grad(y, w) = }')
#print(f'{torch.autograd.grad(l, y) = }')
print(f'Before .backward() ... {w.grad = }')
l.backward() # Populates .grad field of leaf tesnsors (in this case only w)
print(f'After .backward() ... {w.grad = }')

# 5)
# Correct update
w.data = w.data - 0.1 * w.grad
# Incorrect update
#w = w - 0.1 * w.grad
w.grad = None
w.require_grad = True
print(f'after require {w = }')
k = w**2
print(f'{k = }')
k.backward()
print(f'{w = }')
print(f'{w.grad}')



# 6)
w = torch.tensor(1.0, requires_grad=True)
print(f'{w.grad = }')
def loss(w):
    x = torch.tensor(2.0)
    b = torch.tensor(3.0)    
    a = x + b
    y = torch.exp(w)
    l = (y-a)**2
    y/=2 # The computation graph stores "variable versions" and records link to "version 0", whereas after the inplace modification, y is "version 1".
    # This does not happen for y = y / 2
    del y,a,x,b,w # This is redundant since w is local shadowing the outer scope w. del only removes the local variable
    return l
loss(w).backward()
print(f'{w.grad = }')

#import code
#code.interact(local=locals())
code.interact(local = locals())



