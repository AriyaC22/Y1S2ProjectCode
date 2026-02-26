point1 =(4.213E-5, 0.001988)
point2 =(8E-5, 0.003707)

m = (point1[1]-point2[1])/(point1[0]-point2[0])
c = point1[1] - (m*point1[0])

print(m)
print(c)