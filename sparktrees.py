
#https://www.youtube.com/watch?v=CBBV9TcKSqo

#https://stackoverflow.com/questions/45711796/unable-to-start-worker-node-for-apache-spark-2-1-1

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark import SparkConf, SparkContext
from numpy import array

#Boilerplate Spark stuff:
conf = SparkConf().setMaster("local").setAppName("SparkDecisionTree")
sc = SparkContext(conf=conf)

#some functions that convert our cvs input data into numerical features for each candidate

def binary(YN):
    if (YN == 'Y'):
       return 1
    else:
       return 0

def mapEducation(degree)
    if (degree == 'BS'):
       return 1
    elif (degree == 'MS'):
       return 2
    elif (degree == 'PhD'):
       return 3
    else:
       return 0

#convert a list of raw fields from our CSV file to a LabelPoint that MLLib can use. 
#All data must be numerical
def createLabeledPoints(fields):
    yearsExperience = int(fields[0])
    employed = binary(fields[1])
    previousEmployers = binary(fields[2])
    educationLevel = mapEducation(fields[3])
    topTier = binary(fields[4])
    interned = binary(fields[5])
    hired = binary(fields[6])

    return LabeledPoint(hired, array([yearsExperience, employed, previousEmployers,
          educationLevel, topTier, interned]))

#Load up our CSV file, and filter out the header line with the column names
rawData = sc.textFile("trees_test.csv")
header = rawData.first()
rawData = rawData.filter(lambda x: x!= header)

#split each line into a list based on comma delimiters
csvData = rawData.map(lambda x: x.split(","))

#convert these lists to LabeledPoints
trainingData = csvData.map(createLabeledPoints)

#create a test candidate, with 10 years of experience, currenctly employed, 
#3 previous employers, a BS degree, but from a non-top-tier school whhere he or she 
# did not do an internship. You could of course load a huge RDD of test candidates 
#disk too.

testCandidates = [array([10,1,3,1,0,0])]
testData = sc.parallelize(testCandidates)

#Train our DecisionTree classifier using our data set
model = DecisionTree.trainClassifier(trainingData, numClasses=2,
                                      categoricalFeaturesInfo={1:2,3:4,4:2,5:2},
                                      impurity = 'gini', maxDepth=5, maxBins=32)

#Now get predictions for unknown candidates.(Note, you could separate the source
#data into training set and a test set while tuning parameters and measure accuracy as you go!)

predictions = model.predict(testData)
print('Hire prediction:')
results = predictions.collect()
for result in results:
    print result


#we can also print out the decision tree itself
print('Learned classification tree model:')
print(model.toDebugString())



