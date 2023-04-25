import os

# path = "data"
# datanames = os.listdir(path)
# for i in datanames:
#     if i == "defect" or i == "non-defect" or i == "label.txt":
#         pass
#     else:
#         print(i)
#         new_lines = []
#         with open("data/" + i, "r") as f:
#             lines = f.readlines()  #
#             for line in lines:
#                 new_line = line.split("data")
#                 for j in range(len(new_line)):
#                     if j == 2:
#                         print (new_line[j])
#                         new_lines.append('.'+new_line[j])
#         print(lines)
#         print(new_lines)
#         #
#
#         with open("data/" + i, "w") as f:
#             for new_line in new_lines:
#                 f.writelines(new_line)
#
path = "data"
datanames = os.listdir(path)
for i in datanames:
    if i == "defect" or i == "non-defect" or i == "label.txt":
        pass
    else:
        print(i)
        new_lines = []
        with open("data/" + i, "r") as f:
            lines = f.readlines()  #
            for line in lines:
                new_line = line.replace('\\','/')
                new_lines.append(new_line)
        print(lines)
        print(new_lines)
        #
        with open("data/" + i, "w") as f:
            for new_line in new_lines:
                f.writelines(new_line)



# postive_data = []
# for i in datanames:
#     if i == "origin_data":
#         datanames_good = os.listdir("data/" + i)
#     elif i == "process_data":
#         datanames_bad = os.listdir("data/" + i)
# for i in datanames_good:
#     name = os.listdir("data/origin_data/" + i)
#     postive_data.append(name[1] + ' ' + '0')
#     print(i)
# for i in datanames_bad:
#     name = os.listdir("data/process_data/" + i)
#     print(i)
# print(postive_data)
