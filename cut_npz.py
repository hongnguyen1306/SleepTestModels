import numpy as np
import os

data = np.load('/home/rosa/TestModels/SC4072E0.npz', allow_pickle=True)
y = data['y']
x = data['x']

# Lấy tất cả các thuộc tính trong file npz
# In tất cả các thuộc tính

print("x ", len(x))

indices = np.where(y == 3)[0]
print(indices)
print(np.where(y == 4)[0])


# y_3 = y[y == 3]
# y_selected = y[y == 4]
# print("y_3 ", y_3)
# print("y_selected ", y_selected)
# print("x[1:2] ", len(x[400:410])) good

print("x[1:2] ", len(x[911:912]))
print("y_selected[:1] ", y[911:912])

save_dict = {
            "x": x[911:912],
            "y": y[911:912],
            "fs": data["fs"],
            "ch_label": data["ch_label"],
            "header_raw": data["header_raw"],
            "header_annotation": data["header_annotation"],
        }

# # x_selected = x[y == 4]
np.savez('/home/rosa/TestModels/data/test_data.npz', **save_dict)