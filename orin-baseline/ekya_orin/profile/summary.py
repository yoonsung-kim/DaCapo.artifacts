import csv

MODE_LIST = ["default", "30W", "15W"]

if __name__ == "__main__":
    stu_writer = csv.writer(open("student.csv", 'w'))
    stu_writer.writerow(["model", "watt", "inference", "train"])
    
    tea_writer = csv.writer(open("teacher.csv", 'w'))
    tea_writer.writerow(["model", "watt", "inference"])
    
    for mode in MODE_LIST:
        stu_reader = csv.reader(open(f"{mode}-student.csv", 'r'))
        next(stu_reader)
        
        for stu_info in stu_reader:
            stu_writer.writerow([stu_info[0], mode, stu_info[1], stu_info[2]])
        
        tea_reader = csv.reader(open(f"{mode}-teacher.csv", 'r'))
        next(tea_reader)
        
        for tea_info in tea_reader:
            tea_writer.writerow([tea_info[0], mode, tea_info[1]])
