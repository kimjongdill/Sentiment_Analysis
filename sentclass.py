import sys
import pandas
from classifier import Linear_SVM

# Read the arg entry


# if __name__ == "__main__":
#     if len(sys.argv) != 3:
#         print("Usage: python3 sentclass.py file sheetname")
#         exit(0)
#
#     filename = sys.argv[1]
#     sheetname = sys.argv[2]
#     try:
#         inputFile = pandas.read_excel(filename, header=0, sheet_name=sheetname)
#     except:
#         print("Could not open file: " + filename + " Sheet: " + sheetname)
#
#     training_tweets =