
import os
ignore_list = ['defect','non-defect','label.txt','background','origin']
path = "data"
datanames = os.listdir(path)
for i in datanames:
    if i in ignore_list:
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