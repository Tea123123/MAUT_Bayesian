import Simulation as s

a_list = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

t1 = [1,1,1]
t2 = [3,1,1]
t3 = [5,1,1]

w1 = [1/3, 1/3, 1/3]
w2 = [0.5, 0.25, 0.25]
w3 = [0.8, 0.1, 0.1]

t = [t1, t2, t3]
w = [w1, w2, w3]

para_list = [(tt, ww) for tt in t for ww in w]

i=0
for para in para_list:
    t = para[0]
    w = para[1]
    
    temp_d = {}
    
    for a in a_list:
        s = s.Simulation(n_alt=5, n_attr=3, w=w, a=[a,0,0], mean=[0,0,0], std=[5,5,5], ea = t)
        s.run(10000)
        s.save_output("exp1/" + str(i))
        
    i+=1