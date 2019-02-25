import os
import json
import xml.etree.ElementTree
import csv
import datetime
import argparse


"""
outPath = "./sample_img/out"
labelPath = "./test/training/annotations"
logfilePath = "./evaluation_log.csv"
checkpointPath = "./ckpt/checkpoint"

imgdirParameter = "sample_img/"
modelParameter = "cfg/yolo_test.cfg"
loadParameter = "-1"

accuracyMethod = 1
"""


def getOverlappingPercentage(rectA, rectB):
    overlappingX = min(rectA[2], rectB[2]) - max(rectA[0], rectB[0])
    overlappingY = min(rectA[3], rectB[3]) - max(rectA[1], rectB[1])

    rectAX = rectA[2] - rectA[0]
    rectAY = rectA[3] - rectA[1]
    rectAArea = rectAX * rectAY

    rectBX = rectB[2] - rectB[0]
    rectBY = rectB[3] - rectB[1]
    rectBArea = rectBX * rectBY

    if (overlappingX >= 0) and (overlappingY >= 0):
        overlappingArea = overlappingX * overlappingY
    else:
        overlappingArea = 0

    rectArea = max(rectAArea, rectBArea)
    return overlappingArea / rectArea


def main(_outPath, _labelPath, _logfilePath, _checkpointPath, _imgDir, _model, _weights,
         _accuracyMethod="overlappingPercentage"):

    os.chdir("model/darkflow")
    os.system("flow --imgdir " + "../../"+_imgDir + " --model " + _model + " --load " + _weights + " --json --gpu 0.9")
    os.chdir("../../")
    # parse json out
    jsonFilePaths = [os.path.join(_outPath, fileName) for fileName in os.listdir(_outPath)
                     if os.path.isfile(os.path.join(_outPath, fileName))]

    # parse xml label
    xmlFilePaths = [os.path.join(_labelPath, fileName) for fileName in os.listdir(_labelPath)
                    if os.path.isfile(os.path.join(_labelPath, fileName))]

    testObjectsNumber = 0
    evaluateRects = []

    labelObjectsNumber = 0
    labelRects = []
    evaluateRect = []
    imageAccuracy = []
    falseNegativesList = []
    falsePositivesList = []
    truePositivesList = []

    # loop through each image (label + predictions)
    for xmlFilePath, jsonFilePath in zip(xmlFilePaths, jsonFilePaths):
        # json parsing
        with open(jsonFilePath, "r") as fileHandler:
            jsonObjects = json.load(fileHandler)
            jsonObject = []
            for jsonObject in jsonObjects:
                xmin = jsonObject["topleft"]["x"]
                ymin = jsonObject["topleft"]["y"]
                xmax = jsonObject["bottomright"]["x"]
                ymax = jsonObject["bottomright"]["y"]
                evaluateRects.append([xmin, ymin, xmax, ymax])
            testObjectsNumber += len(jsonObject)

        # xml parsing
        annotation = xml.etree.ElementTree.parse(xmlFilePath).getroot()
        xmlObjects = annotation.findall("object")
        xmlObject = []

        for xmlObject in xmlObjects:
            xmlBndbox = xmlObject.find("bndbox")
            xmin = int(xmlBndbox.find("xmin").text)
            ymin = int(xmlBndbox.find("ymin").text)
            xmax = int(xmlBndbox.find("xmax").text)
            ymax = int(xmlBndbox.find("ymax").text)
            labelRects.append([xmin, ymin, xmax, ymax])
        labelObjectsNumber += len(xmlObject)

        # calculate accuracy and false|true positives|negatives
        falseNegatives = 0
        falsePositives = 0
        truePositives = 0
        accuracies = []

        for i in range(max(len(labelRects), len(evaluateRects))):
            bestAccuracy = 0
            bestLabelIndex = None
            bestEvaluateIndex = None
            for labelRectIndex, labelRect in enumerate(labelRects):
                # bestAccuracy = 0
                for evaluateRectIndex, evaluateRect in enumerate(evaluateRects):
                    singleAccuracy = getOverlappingPercentage(evaluateRect, labelRect)

                    if singleAccuracy > bestAccuracy:
                        bestAccuracy = singleAccuracy
                        bestEvaluateIndex = evaluateRectIndex
                        bestLabelIndex = labelRectIndex
                # accuracies.append(singleAccuracy)
            # entering this if statement if regions not overlap
            if bestEvaluateIndex is None:
                # remaining label regions
                # add 0 accuracy for each remaining label region
                accuracies.extend([0 for _ in range(len(labelRects))])
                # all remaining label regions count as false negatives because they were not found
                falseNegatives += len(labelRects)

                # remaining evaluation regions
                # add 0 accuracy for each remaining evaluation region
                accuracies.extend([0 for _ in range(len(evaluateRects))])
                # all remaining evaluation regions count as false positives because they were found by mistake
                falsePositives += len(evaluateRects)

                break
            else:
                # always take best matching pair (defined by best overlapping) and delete regions from list
                accuracies.append(bestAccuracy)
                del(evaluateRects[bestEvaluateIndex])
                del(labelRects[bestLabelIndex])
                # every overlapping image is counted as true positive
                # TODO: minimum of 0.5 accuracy requirement for true positive could be added
                # this commented out code could conflict with the logic of the overlapping percentage calculation
                # if bestAccuracy >= 0.5:
                #     truePositives += 1
                # else:
                #     falseNegatives += 1
                truePositives += 1

            #print("accuracies", accuracies)

        # no object in the image
        else:
            print("else")
            accuracies = [1]

        # calculate accuracy for whole image
        imageAccuracy.append(sum(accuracies) / len(accuracies))

        # add binary classificators to list
        truePositivesList.append(truePositives)
        falseNegativesList.append(falseNegatives)
        falsePositivesList.append(falsePositives)

    accuracy = None
    if _accuracyMethod == "objectNumber":
        accuracy = testObjectsNumber / labelObjectsNumber
    elif _accuracyMethod == "overlappingPercentage":
        accuracy = sum(imageAccuracy) / len(imageAccuracy)

    print("Accuracy = " + str(accuracy))

    if not os.path.exists(_logfilePath):
        with open(_logfilePath, "w"):
            pass

    # compare result
    with open(_logfilePath, "r+") as fileHandler:
        bestAccuracy = 0
        csvReader = csv.reader(fileHandler)
        csvWriter = csv.writer(fileHandler, delimiter=",",
                               quotechar="\"", quoting=csv.QUOTE_NONNUMERIC)
        header = next(csvReader, "")
        if header is "":
            csvWriter.writerow(["date", "accuracy", "true positives", "false positives", "false negatives", "improved"])
        else:
            for row in csvReader:
                # if not row:
                if float(row[1]) > bestAccuracy:
                    bestAccuracy = float(row[1])

        # calculate binary classificators average
        truePositivesValue = sum(truePositivesList) / len(truePositivesList)
        falseNegativesValue = sum(falseNegativesList) / len(falseNegativesList)
        falsePositivesValue = sum(falsePositivesList) / len(falsePositivesList)

        # insert new value
    #    csvWriter.writerow([datetime.datetime.now(), accuracy, truePositivesValue, falsePositivesValue,
    #                        falseNegativesValue])

        print("ACCURACY:", accuracy)
        print("BEST ACCURACY:", bestAccuracy)
        # funktioniert so nicht, weil checkpoints schon "neu" sind
        # new best accuracy
        if accuracy > bestAccuracy:
            print("MODEL IMPROVED")
            os.system("rm " + _checkpointPath + "_tmp")
            csvWriter.writerow([datetime.datetime.now(), accuracy, truePositivesValue, falsePositivesValue,
                                falseNegativesValue, "yes"])

        # old accuracy is better
        else:
            print("MODEL NOT IMPROVED.")
            os.system("mv " + _checkpointPath + "_tmp" + " " + _checkpointPath)
            csvWriter.writerow([datetime.datetime.now(), accuracy, truePositivesValue, falsePositivesValue,
                                falseNegativesValue, "no"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Face Detection.")

    parser.add_argument("--outPath", required=True, type=str,
                        help="Path to JSON out files.")
    parser.add_argument("--labelPath", required=True, type=str,
                        help="Path to label files.")
    parser.add_argument("--logfilePath", required=True, type=str,
                        help="Log file path (filename included).")
    parser.add_argument("--checkpointPath", required=True, type=str,
                        help="Checkpoint file path (filename included).")
    parser.add_argument("--imgDir", required=True, type=str,
                        help="Path to image files.")
    parser.add_argument("--model", required=True, type=str,
                        help="Path to model config file (filename included).")
    parser.add_argument("--weights", required=True, type=str,
                        help="Path to weight file (filename included).")
    parser.add_argument("--accuracyMethod", choices=["objectNumber", "overlappingPercentage"],
                        default="overlappingPercentage", required=False, type=str,
                        help="Method for calculating accuracy.")

    args = parser.parse_args()

    outPath = args.outPath
    labelPath = args.labelPath
    logfilePath = args.logfilePath
    checkpointPath = args.checkpointPath
    imgDir = args.imgdir
    model = args.model
    weights = args.weights
    accuracyMethod = args.accuracyMethod

    main(args.outPath, args.labelPath, args.logfilePath, args.checkpointPath, args.imgDir, args.model, args.weights,
         args.accuracyMethod)
