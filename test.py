import time
from TestAlgo import createData, save_obj, load_obj

createData('GOOG')
data = load_obj('data/Goog')

t = float("{:.6f}".format(time.time()))
for number, value in enumerate(data['Open'][0:len(data['Open'])-4]):
	sum(data['Open'][number: number+4])
elapsed = "{:.6f}".format(time.time()-t)
print(elapsed)
