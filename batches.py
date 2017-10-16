
# suppose we have
import cPickle as pickle
import numpy as np
print(np.random.rand(2))

array = []
for i in range(1724):
    array.append(np.random.rand(2))

# of groups per groop

sample_per_batch = 50
num_batches = int(len(array)/50)
residual_batch = len(array)%50

k_ = np.split(np.array(array[:-1*residual_batch]),num_batches)

last_batch = array[-1*sample_per_batch:]

k = np.concatenate((k_,[last_batch]),axis = 0)

returndata_loaded

num_epoch = 50

epoch_store = []
epoch_store_index = 0
for i,j in enumerate(k):
    # now k has a name and a time stamp.

    one_batch_x = []
    one_batch_target = []
    for ik,jk in enumerate(j):
        date = jk[0]
        stock_name = jk[1]
        next_day = ...
        with open( ...,'rb') as f:
            stock_data = pickle.load(f)
        desired_date_data = stock_data[date]

        target_return = return_data[next_day]

        one_batch_x.append(desired_date_data)
        one_batch_target.append(target_return)

    epoch_store.append([one_batch_x,one_batch_target])

    if len(epoch_store)%num_epoch == 0 and len(epoch_store)!=0:

        with open('epoch'+str(epoch_store_index),'wb') as f:
            pickle.dump(epoch_store,f)
            epoch_store_index = epoch_store_index + 1

    if i == len(k)-1:
        with open('epoch'+str(epoch_store_index),'wb') as f:
            pickle.dump(epoch_store,f)
            epoch_store_index = epoch_store_index + 1


# we could write in a manner of files.