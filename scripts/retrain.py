import os
import subprocess
import configparser
import datetime
import csv


def checkdata(path, ext):
    with os.scandir(path) as it:
        for entry in it:
            if not entry.name.startswith(".") and entry.is_file():
                end = os.path.basename(entry.name).rsplit(".", 3)
                if end[1] == ext:
                    print("{0} is {1}".format(end[0], ext))
                else:
                    if ext == "xml":
                        filename = os.environ["HOME"]\
                                   + "/retrain_evaluate_prozesskette/new_data/annotations/" + entry.name
                        os.remove(filename)
                    else:
                        filename = os.environ["HOME"] + "/retrain_evaluate_prozesskette/new_data/images/" + entry.name
                        os.remove(filename)

def main():
    path_img = "new_data/images/"
    path_ann = "new_data/annotations/"
    ext = ["jpg", "xml"]

    config = configparser.ConfigParser()
    config.read("./config.cfg")
    retrain_command = config.get("commands", "run_training")

#    checkdata(path_img, ext[0])
#    checkdata(path_ann, ext[1])

    print("cp " + "./model/darkflow/ckpt/checkpoint" + " " + "./model/darkflow/ckpt/checkpoint" + "_tmp")
    os.system("cp " + "./model/darkflow/ckpt/checkpoint" + " " + "./model/darkflow/ckpt/checkpoint" + "_tmp")

    #Startzeit
    t_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    os.chdir("model/darkflow")
    subprocess.call([retrain_command], shell=True)
    os.chdir("../../")

    #Endzeit
    t_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(os.getcwd())
    from .evaluate import main as ev_main
    ev_main("./test_data/images/out/", "./test_data/annotations", "./evaluate_log.csv",
            "./model/darkflow/ckpt/checkpoint", "./test_data/images/", "./cfg/yolo_test.cfg", "-1")

    with open("/home/julius/retrain_evaluate_prozesskette/time.csv", "r+") as fileHandler:
        csvReader = csv.reader(fileHandler)
        csvWriter = csv.writer(fileHandler, delimiter=",",
                               quotechar="\"", quoting=csv.QUOTE_NONNUMERIC)
        header = next(csvReader, "")
        if header is "":
            csvWriter.writerow(["start", "stop"])
        csvWriter.writerow([t_start, t_end])

if __name__ == "__main__":
    main()
