#  Alex Hobeychi
## The purpose of this program is to show the effectiveness of the A* start algorithm in finding the optimal path between different points
# Necessary packages:
## - geopandas
## - pandas
## - matplotlib: pyplot, ticker
## - numpy
## - seaborn
## These are all installed by default by anaconda

import geopandas as gpd
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
import seaborn as sns

crimeGeoPanda = gpd.read_file('../data/crime_dt.shp')
df = pd.DataFrame(crimeGeoPanda)


def calcuateNumberOfBins(min, max, size):
    return int(abs( (max - min) // size))





xmin, xmax = -73.55, -73.59
ymin, ymax = 45.49, 45.53

def getGridSize():
    return float(input('Input Grid Size: '))


gridSize = getGridSize()

numberOfBins = calcuateNumberOfBins(xmax,xmin, gridSize) # This is the number of groups/blocks on each axis
print(numberOfBins)
xwidth = (xmax - xmin)/numberOfBins
ywidth = (ymax - ymin)/numberOfBins



def getX_Value(obj):
    return obj.x
    
def getY_Value(obj):
    return obj.y

df['X_VALUE'] = df['geometry'].apply(getX_Value)
df['Y_VALUE'] = df['geometry'].apply(getY_Value)    


# Now, we have to convert the geometry Column to more usable features, seperate x,y and then well put them into a tuple to make the rest easier


def getXGroup(value):
    return (value - xmin) // xwidth
def getYGroup(value):
    return (value - ymin) // ywidth

df['X GROUP'] = df['X_VALUE'].apply(getXGroup)
df['Y GROUP'] = df['Y_VALUE'].apply(getYGroup)
df['X-Y GROUP'] = (df['X GROUP'].astype(str) + "," + df['Y GROUP'].astype(str))
df.head()


# Now we will create a list of these X-Y groups so that we can use their indexes later to store the total amount of occurences in a 2d array


def convertToList(str):
    x, y = str.split(',')
    x  = float(x)
    y  = float(y)
    return (int(x),int(y))

listOfOccurences = list(df['X-Y GROUP'].apply(convertToList))
arrayOfOccurences = np.zeros((numberOfBins,numberOfBins))

for element in listOfOccurences:
    x, y = element
    arrayOfOccurences[(numberOfBins - 1) - y,(numberOfBins - 1) - x] += 1


## Now that we have all the data stored in a numpy Array, we can calculate the mean, standard deviation of the total


mean = np.mean(arrayOfOccurences.flatten())
print('The mean of the crimes per area for ' + str(numberOfBins) + 'x'+ str(numberOfBins)  + ' areas is ' + str(mean))
std = np.std(arrayOfOccurences.flatten())
print('The standard deviation of the crimes per area for ' + str(numberOfBins) + 'x'+ str(numberOfBins)  + ' areas is ' + str(std))


# Now, the dataframe and the 2d Array are set up lets visualize the dataframe
# We'll clean up the axis 


xs = np.linspace(xmin,xmax,numberOfBins)
ys = np.linspace(ymax, ymin, numberOfBins)
x_AXIS = []
y_AXIS = []
for x, y in zip(xs,ys):
    x_AXIS.append('{:.3f}'.format(x))
    y_AXIS.append('{:.3f}'.format(y))



data2 = pd.DataFrame(data=arrayOfOccurences, index = y_AXIS, columns= x_AXIS )



fig, ax = plt.subplots(1, figsize=(18, 15))
sns.heatmap(data=data2, cmap ='plasma', linewidth=1, cbar= True, annot= True)
ax.set_title('Total Number Of Crimes per Area', fontsize= 20)
plt.show()

## Above is the Graph of the total area, later I will use another method to make it easier to plot the line as a heatmap won't allow to draw a line.
## Set up dataFrames with all the threshold information

threshold = int(input('input the threshold: '))   
thresholdValue = np.percentile(arrayOfOccurences.flatten(),threshold)




def thresholdTransformation(num):
    if num < thresholdValue:
        return 0
    return num



arrayCrime = np.array(arrayOfOccurences)

x, y = arrayOfOccurences.shape

for i in range(x):
    for j in range(y):
        arrayCrime[i][j] = thresholdTransformation(arrayCrime[i][j])
        
print('For the matrix with crime with threshold {}, the std is {} and the mean is {}'.format\
      (thresholdValue,np.std(arrayCrime),np.mean(arrayCrime)))

arrayCrime = (arrayCrime != 0)
dfThreshold = pd.DataFrame(data=arrayCrime, index = y_AXIS, columns= x_AXIS )



ax = sns.heatmap(data=dfThreshold, cmap ='viridis', linewidth=1.5, cbar= False, annot= False, linecolor='black')
sns.despine(offset=10, trim=True)
sns.set(style="whitegrid")
plt.title('Map with threshold {}'.format(threshold), fontsize = 20) # title with fontsize 20
plt.tight_layout()
plt.show()

# PathFinding 
# Let us graph a raw matlplot that will allow us to easily graph the path line
# First we have to flip the arrays that hold the threshold information to make the plots fit the way we want them to.


arrayCrimeFlip = np.flipud(arrayCrime)

# A* Algorithm Implementation
# First We can define all the posssible movements on the map


movement = [(1,0),(0,1),(-1,0),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]


# Easily Convert longitute and latitude to fit within the array


def getTile(x,y):
    ceildiv = lambda a,b: -(-a // b)
    xTile = ceildiv((x-xmin),xwidth)
    yTile = ceildiv((y-ymin),ywidth)
    return xTile, yTile


### Calculate the weight of a given movement given the following conditions: 
#### 1) movement allong a colored tile is worth 1.3. 
#### 2) diagonal movement is worth 1.5. 
#### 3) free movement not along a color block is worth 1.

def getPathWeight(start, end, array):

    x1, x2 = start[0], end[0]
    y1, y2 = start[1], end[1]
    transition = (x2 - x1, y2 - y1)

    # Up
    if(transition == (0.0, 1.0)):
        try:
            return 1.3 if (array[y1][x1] == 1 or array[y1][x1 - 1] == 1) else 1
        except:
            return  1.3 if array[y1][x1] == 1 else  1

    # Down
    elif(transition == (0.0, -1.0)):
        try:
            return  1.3 if (array[y1 - 1][x1] == 1 or array[y1 - 1][x1 - 1] == 1) else 1
        except:
            return  1.3 if (array[y1 - 1][x1] == 1) else 1

    # Right
    elif(transition == (1.0, 0)):
        try:
            return  1.3 if (array[y1][x1] == 1 or array[y1-1][x1] == 1) else 1
        except:
            return 1.3 if (array[y1][x1] == 1) else 1
            
    # Left
    elif(transition == (-1.0, 0)):
        try:
            return 1.3 if (array[y1][x1 - 1] == 1 or array[y1 - 1][x1 - 1] == 1) else 1
        except:
            return 1.3 if (array[y1][x1 - 1] == 1) else 1

    else:
        return 1.5


### We define a class that will hold all necessary information to process the computation


class node:

    def __init__(self, previous, location):
        self.previous = previous
        self.location = location
        self.x = location[0]
        self.y = location[1]

        self.gn = 0
        self.hn = 0
        self.fn = 0

    def __eq__(self, other):
        return self.location == other.location

    def __lt__(self, other):
        return self.fn < other.fn

    def __gt__(self, other):
        return self.gn > other.gn

    def createNewLocation(self,shift):
        return (self.location[0] + shift[0], self.location[1] + shift[1])

    def getLocation(self):
        return (self.x, self.y)

    def calculateValues(self, other, end, array):
        #actual cost of path from start to node n
        weight = getPathWeight(self.getLocation(), other.getLocation(), array)
        self.gn = other.gn + weight
        # estimate of cost to reach goal from node n 
        # Uses Manhattan Distance
        self.hn = abs(self.x - end.x) + abs(self.y - end.y)
        self.previous.hn = self.hn + weight
        # Uses Pythagorean Distance, Seems to improve performance
        # self.hn = (abs(self.x - end.x)**2 + abs(self.y - end.y)**2)**0.5
        
        # estimate of total cost along path through n
        self.fn = self.gn + self.hn
        
    def __str__(self):
        return (str((x,y)))


### We define all restrictions for the pathfinding algorithm
# checking if the area is valid (not surrounded by blocked areas)


def invalidArea(array, coord):

    current = (array[coord[1]][coord[0]] == 1) 
    left = (array[coord[1]][coord[0] - 1] == 1) 
    bottom = (array[coord[1] - 1][coord[0]] == 1) 
    bottomLeft = (array[coord[1] - 1][coord[0] - 1] == 1)
    
    return current and left and bottom and bottomLeft


# checking if the area selecting is within the boundaries of the board


def withinBoundary(location):
    x, y = location[0], location[1]
    return (x < numberOfBins and y < numberOfBins) and (x > 0 and y > 0)


# checking if a movement from a given area is valid 


def invalidMove(coord, shift, array):

    newLocation = (coord[1] + shift[1], coord[0] + shift[0])
    if not withinBoundary(newLocation):
        return True
        
    # (1,1) Right Up
    if shift == (1,1):
        return (array[coord[1]][coord[0]] == 1) 

    # (-1,1) Left Up
    elif shift == (-1,1):
        return (array[coord[1]][coord[0] - 1] == 1) 

    # (-1,-1) Down Left
    elif shift == (-1,-1):
        return (array[coord[1] - 1][coord[0] - 1] == 1)

    # (1,-1) Down Right
    elif shift == (1,-1):
        return (array[coord[1] - 1][coord[0]] == 1)

    # (1, 0) Right
    elif shift == (1,0):
        try:
            return (array[coord[1]][coord[0]] == 1) and (array[coord[1] - 1][coord[0]] == 1)
        except:
            return (array[coord[1]][coord[0]] == 1)
            
    # (-1, 0) Left 
    elif shift == (-1,0):
        try:
            return (array[coord[1]][coord[0] - 1] == 1 and array[coord[1] - 1][coord[0] - 1] == 1)
        except:
            return (array[coord[1]][coord[0] - 1] == 1)

    # (0, 1) Up
    elif shift == (0,1):
        try:
            return (array[coord[1]][coord[0]] == 1 and array[coord[1]][coord[0] - 1] == 1)
        except:
            return (array[coord[1]][coord[0]] == 1)

    # (0,-1) Down 
    elif shift == (0,-1):
        try:
            return (array[coord[1] - 1][coord[0]] == 1 and array[coord[1] - 1][coord[0] - 1] == 1)
        except:
            return (array[coord[1] - 1][coord[0]] == 1)
            
    return False 


### Define the bulk Of the A* Algorithm


def search(start, end, array):
    
    # We check wether or not the end and start are valid
    if (not (withinBoundary(start) and withinBoundary(end))) or invalidArea(array, start) or invalidArea(array,end):
        print('The input was not valid')
        return None
    
    # We define the known nodes
    start = node(None, start)
    end = node(None, end)

    # To track all the nodes
    opened, closed = [],[]

    # We start with the first
    opened.append(start)

    while(opened):
        
        current = opened[0]
        index = 0
        
        for i, test in enumerate(opened):
            if test < current:
                current = test
                index = i

        # We remove from open and add it to close
        opened.pop(index)
        closed.append(current)

        # We See if we're done
        if current == end:
            path = []
            temp = current
            while (temp):
                path.append(temp.location)
                temp = temp.previous 
            return path, current.gn

        # We explore all the possible neighbours
        neighbours = []
        for shifts in movement:
            
            newLocation = current.createNewLocation(shifts)

            # Check if the new location is valid
            if (not withinBoundary(newLocation)) or invalidMove(current.getLocation(), shifts, array):
                continue

            newNode = node(current, newLocation)

            # check that the node wasn't already visited to remove redundancies 
            # if(newNode in closed):
                # for x in closed:
                    # if newNode == x and newNode.gn > x.gn:
                        # continue
            if(newNode in closed):
                continue
            else:
                neighbours.append(newNode)

        for neighbor in neighbours:
            
            # Now we check that we havent visited the new location
            for closedNode in closed:
                if neighbor == closedNode:
                    continue
            
            # We set the values of g(n), h(n) and f(n)
            neighbor.calculateValues(current,end,array)

            # neighbor is already in the open list
            for open in opened:
                if neighbor == open and neighbor > open:
                    continue

            # Add the child to the open list
            opened.append(neighbor)

### Define all the necessary plotting tools and funtions

from time import time

def unzipPath(path):
    x,y  = [], []
    for j in path:
        x.append(j[0] - .5)
        y.append(j[1] - .5)
    return np.array(x),np.array(y)

def plot_path(start, end):
    __draw(start,end,arrayCrimeFlip)

def __draw(start, end, array):

    startTime = time()
    searchResults = search(start,end, array)

    if(searchResults == None or len(searchResults[0]) == 1 or start == end):
        print('no route exists')
        return None

    endTime = time()
    realPath = searchResults[0][::-1]

    if(len(realPath) == 0):
        return
    
    x, y = unzipPath(realPath)
    weight = searchResults[1]

    titleWithRunTime = 'Pathfinding - Process Time: {} seconds, Path Weight: {}'.format(round((endTime-startTime),3), round(weight,1))
    fig, ax = plt.subplots(figsize = (12,12))
    ax.imshow(array, cmap='viridis', zorder=0)
    plt.xlim([0,numberOfBins-1])
    plt.ylim([0,numberOfBins-1])    
    ax.set_xticks(np.arange(-.5, numberOfBins, 1))
    ax.set_yticks(np.arange(-.5, numberOfBins, 1))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%g') % (ymin +   (x * ywidth))))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: ('%g') % (xmin +   (x * xwidth))))
    ax.set_title(titleWithRunTime, fontsize='20')
    sns.despine(offset=20, trim=True)
    sns.set(style="whitegrid")
    plt.tight_layout()
    ax.grid(linewidth=2, color= 'black')
    ax.scatter(x[0],y[0], marker = ".", color = "white", s = 200, zorder=2)
    ax.scatter(x[-1],y[-1], marker = "X", color = "white", s = 400,zorder=2)
    ax.plot(x,y, color = "white", linewidth=3,zorder=1)
    plt.show()


### Testing the algorithm 

def getCoordinates():
    str = input('Input locations separated by a comma so x1,y1,x2,y2: ')
    x1, y1, x2, y2 = str.split(',')
    return int(x1), int(y1), int(x2), int(y2)


x1, y1, x2, y2 = getCoordinates()
start = (x1,y1)
end = (x2,y2)

plot_path(start, end)

print('the program has now terminated')