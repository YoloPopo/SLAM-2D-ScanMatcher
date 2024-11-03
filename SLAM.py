import rasterio
import warnings
import numpy as np
import math, copy, json, os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Tuple, Optional
from rasterio.transform import from_origin

def create_gif_from_images(folder_path, output_path, duration=10):
    # Get a list of all files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the images by name assuming they are named '1', '2', ..., '910'
    image_files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    # Load all images into a list
    images = [Image.open(os.path.join(folder_path, img)) for img in image_files]

    # Save as a GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],  # Append the rest of the images
        duration=duration,         # Duration for each frame (in milliseconds)
        loop=0                     # Loop forever
    )

def convert_to_geotiff(png_path, geotiff_path, top_left=(0, 0), pixel_size=1):
    # Load the PNG image
    image = Image.open(png_path)
    image = image.convert('RGB')  # Ensure the image is RGB
    img_array = np.array(image)

    # Get the dimensions of the image
    height, width, _ = img_array.shape

    # Define geotransform (top-left coordinates, pixel size, and rotation)
    transform = from_origin(top_left[0], top_left[1], pixel_size, pixel_size)

    # Suppress NotGeoreferencedWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # Create a new GeoTIFF file with the same dimensions
        with rasterio.open(
            geotiff_path, 'w',
            driver='GTiff',
            height=height,
            width=width,
            count=3,  # Number of channels (R, G, B)
            dtype=img_array.dtype,
            crs='+proj=latlong',  # Coordinate Reference System
            transform=transform
        ) as dst:
            # Write RGB bands to the GeoTIFF
            dst.write(img_array[:, :, 0], 1)  # Red channel
            dst.write(img_array[:, :, 1], 2)  # Green channel
            dst.write(img_array[:, :, 2], 3)  # Blue channel

    print(f"GeoTIFF created: {geotiff_path}")

class OccupancyGrid:
    def __init__(self, mapXLength, mapYLength, initXY, unitGridSize, lidarFOV, numSamplesPerRev, lidarMaxRange, wallThickness):
        # Initialize grid parameters
        xNum = int(mapXLength / unitGridSize)
        yNum = int(mapYLength / unitGridSize)

        # Set grid x, y coordinates
        x = np.linspace(-xNum * unitGridSize / 2, xNum * unitGridSize / 2, num=xNum + 1) + initXY['x']
        y = np.linspace(-yNum * unitGridSize / 2, yNum * unitGridSize / 2, num=yNum + 1) + initXY['y']
        self.OccupancyGridX, self.OccupancyGridY = np.meshgrid(x, y)

        # Initialize occupancy grid matrices
        self.occupancyGridVisited = np.ones((xNum + 1, yNum + 1))
        self.occupancyGridTotal = 2 * np.ones((xNum + 1, yNum + 1))

        # Set lidar and map parameters
        self.unitGridSize = unitGridSize
        self.lidarFOV = lidarFOV
        self.lidarMaxRange = lidarMaxRange
        self.wallThickness = wallThickness
        self.mapXLim = [self.OccupancyGridX[0, 0], self.OccupancyGridX[0, -1]]
        self.mapYLim = [self.OccupancyGridY[0, 0], self.OccupancyGridY[-1, 0]]
        self.numSamplesPerRev = numSamplesPerRev
        self.angularStep = lidarFOV / numSamplesPerRev
        self.numSpokes = int(np.rint(2 * np.pi / self.angularStep))

        # Calculate spokes grid
        self.xGrid, self.yGrid, self.bearingIdxGrid, self.rangeIdxGrid = self.calculateSpokesGrid()
        self.radByX, self.radByY, self.radByR = self.organizeSpokesGrid()
        self.spokesStartIdx = int(((self.numSpokes / 2 - self.numSamplesPerRev) / 2) % self.numSpokes)

    def calculateSpokesGrid(self):
        """Calculate the grid for lidar spokes."""
        numHalfElem = int(self.lidarMaxRange / self.unitGridSize)
        
        # Create grid with x and y values within lidar max range
        x = np.linspace(-self.lidarMaxRange, self.lidarMaxRange, 2 * numHalfElem + 1)
        y = np.linspace(-self.lidarMaxRange, self.lidarMaxRange, 2 * numHalfElem + 1)
        xGrid, yGrid = np.meshgrid(x, y)
        
        # Calculate bearing index grid based on angle
        bearingIdxGrid = np.zeros((2 * numHalfElem + 1, 2 * numHalfElem + 1))
        bearingIdxGrid[:, numHalfElem + 1:] = np.rint((np.pi / 2 + np.arctan(yGrid[:, numHalfElem + 1:] / xGrid[:, numHalfElem + 1:]))
                                                      / np.pi / 2 * self.numSpokes - 0.5).astype(int)
        bearingIdxGrid[:, :numHalfElem] = np.fliplr(np.flipud(bearingIdxGrid))[:, :numHalfElem] + int(self.numSpokes / 2)
        bearingIdxGrid[numHalfElem + 1:, numHalfElem] = int(self.numSpokes / 2)
        
        # Calculate range index grid
        rangeIdxGrid = np.sqrt(xGrid ** 2 + yGrid ** 2)
        
        return xGrid, yGrid, bearingIdxGrid, rangeIdxGrid

    def organizeSpokesGrid(self):
        """Organize the spokes into radial sections by bearing index."""
        radByX, radByY, radByR = [], [], []

        for i in range(self.numSpokes):
            idx = np.argwhere(self.bearingIdxGrid == i)
            radByX.append(self.xGrid[idx[:, 0], idx[:, 1]])
            radByY.append(self.yGrid[idx[:, 0], idx[:, 1]])
            radByR.append(self.rangeIdxGrid[idx[:, 0], idx[:, 1]])

        return radByX, radByY, radByR

    def expandOccupancyGrid(self, expandDirection):
        """Expand the occupancy grid in the given direction."""
        gridShape = self.occupancyGridVisited.shape

        if expandDirection == 1:
            self._expandGridHelper(0, axis=1)  # Left
        elif expandDirection == 2:
            self._expandGridHelper(gridShape[1], axis=1)  # Right
        elif expandDirection == 3:
            self._expandGridHelper(0, axis=0)  # Bottom
        else:
            self._expandGridHelper(gridShape[0], axis=0)  # Top

    def _expandGridHelper(self, position, axis):
        """Helper method to insert additional space when expanding the grid."""
        gridShape = self.occupancyGridVisited.shape

        if axis == 0:
            insertion = np.ones((int(gridShape[0] / 5), gridShape[1]))
            y = self._calculateNewAxisLimits(position, gridShape, axis)
            xv, yv = np.meshgrid(self.OccupancyGridX[0], y)
        else:
            insertion = np.ones((gridShape[0], int(gridShape[1] / 5)))
            x = self._calculateNewAxisLimits(position, gridShape, axis)
            xv, yv = np.meshgrid(x, self.OccupancyGridY[:, 0])

        # Insert new grid space
        self.occupancyGridVisited = np.insert(self.occupancyGridVisited, [position], insertion, axis=axis)
        self.occupancyGridTotal = np.insert(self.occupancyGridTotal, [position], 2 * insertion, axis=axis)
        self.OccupancyGridX = np.insert(self.OccupancyGridX, [position], xv, axis=axis)
        self.OccupancyGridY = np.insert(self.OccupancyGridY, [position], yv, axis=axis)

        # Update map limits
        self.mapXLim = [self.OccupancyGridX[0, 0], self.OccupancyGridX[0, -1]]
        self.mapYLim = [self.OccupancyGridY[0, 0], self.OccupancyGridY[-1, 0]]

    def _calculateNewAxisLimits(self, position, gridShape, axis):
        """Calculate new axis limits for expansion."""
        if axis == 0:  # y-axis
            if position == 0:
                return np.linspace(self.mapYLim[0] - int(gridShape[0] / 5) * self.unitGridSize,
                                   self.mapYLim[0], num=int(gridShape[0] / 5), endpoint=False)
            else:
                return np.linspace(self.mapYLim[1] + self.unitGridSize,
                                   self.mapYLim[1] + (int(gridShape[0] / 5)) * self.unitGridSize,
                                   num=int(gridShape[0] / 5), endpoint=False)
        else:  # x-axis
            if position == 0:
                return np.linspace(self.mapXLim[0] - int(gridShape[1] / 5) * self.unitGridSize,
                                   self.mapXLim[0], num=int(gridShape[1] / 5), endpoint=False)
            else:
                return np.linspace(self.mapXLim[1] + self.unitGridSize,
                                   self.mapXLim[1] + (int(gridShape[1] / 5)) * self.unitGridSize,
                                   num=int(gridShape[1] / 5), endpoint=False)

    def convertRealXYToMapIdx(self, x, y):
        """Convert real-world coordinates to map index."""
        xIdx = (np.rint((x - self.mapXLim[0]) / self.unitGridSize)).astype(int)
        yIdx = (np.rint((y - self.mapYLim[0]) / self.unitGridSize)).astype(int)
        return xIdx, yIdx

    def checkMapToExpand(self, x, y):
        """Check whether the map needs to expand based on x, y coordinates."""
        if any(x < self.mapXLim[0]):
            return 1
        elif any(x > self.mapXLim[1]):
            return 2
        elif any(y < self.mapYLim[0]):
            return 3
        elif any(y > self.mapYLim[1]):
            return 4
        return -1

    def checkAndExpandOG(self, x, y):
        """Check and expand occupancy grid if necessary."""
        expandDirection = self.checkMapToExpand(x, y)
        while expandDirection != -1:
            self.expandOccupancyGrid(expandDirection)
            expandDirection = self.checkMapToExpand(x, y)

    def updateOccupancyGrid(self, reading, dTheta=0, update=True):
        """Update occupancy grid based on lidar reading and angle."""
        
        # Extract x, y, theta, and range from reading
        x, y, theta, rMeasure = reading['x'], reading['y'], reading['theta'], reading['range']
        
        # Update theta by adding dTheta
        theta += dTheta
        
        # Convert rMeasure to a numpy array for easier processing
        rMeasure = np.asarray(rMeasure)
        
        # Compute the offset index for spokes based on theta
        spokesOffsetIdxByTheta = int(np.rint(theta / (2 * np.pi) * self.numSpokes))
        
        # Initialize empty and occupied cell lists
        emptyXList, emptyYList, occupiedXList, occupiedYList = [], [], [], []

        # Iterate over each sample per revolution
        for i in range(self.numSamplesPerRev):
            # Calculate the spoke index, wrapping around with modulo
            spokeIdx = int(np.rint((self.spokesStartIdx + spokesOffsetIdxByTheta + i) % self.numSpokes))
            
            # Get the direction vectors for x, y, and r at this spoke
            xAtSpokeDir = self.radByX[spokeIdx]
            yAtSpokeDir = self.radByY[spokeIdx]
            rAtSpokeDir = self.radByR[spokeIdx]
            
            # Determine empty and occupied regions
            if rMeasure[i] < self.lidarMaxRange:
                emptyIdx = np.argwhere(rAtSpokeDir < rMeasure[i] - self.wallThickness / 2)
            else:
                emptyIdx = []
            
            occupiedIdx = np.argwhere(
                (rAtSpokeDir > rMeasure[i] - self.wallThickness / 2) & 
                (rAtSpokeDir < rMeasure[i] + self.wallThickness / 2)
            )
            
            # Convert the real world coordinates to map indices
            xEmptyIdx, yEmptyIdx = self.convertRealXYToMapIdx(x + xAtSpokeDir[emptyIdx], y + yAtSpokeDir[emptyIdx])
            xOccupiedIdx, yOccupiedIdx = self.convertRealXYToMapIdx(x + xAtSpokeDir[occupiedIdx], y + yAtSpokeDir[occupiedIdx])
            
            if update:
                # Update the occupancy grid
                self.checkAndExpandOG(x + xAtSpokeDir[occupiedIdx], y + yAtSpokeDir[occupiedIdx])
                
                if len(emptyIdx) != 0:
                    self.occupancyGridTotal[yEmptyIdx, xEmptyIdx] += 1
                
                if len(occupiedIdx) != 0:
                    self.occupancyGridVisited[yOccupiedIdx, xOccupiedIdx] += 2
                    self.occupancyGridTotal[yOccupiedIdx, xOccupiedIdx] += 2
            else:
                # Store empty and occupied cell coordinates if not updating the grid
                emptyXList.extend(x + xAtSpokeDir[emptyIdx])
                emptyYList.extend(y + yAtSpokeDir[emptyIdx])
                occupiedXList.extend(x + xAtSpokeDir[occupiedIdx])
                occupiedYList.extend(y + yAtSpokeDir[occupiedIdx])

        # Return the lists of empty and occupied cell coordinates if update is False
        if not update:
            return np.asarray(emptyXList), np.asarray(emptyYList), np.asarray(occupiedXList), np.asarray(occupiedYList)

class ScanMatcher:
    def __init__(self, og, searchRadius, searchHalfRad, scanSigmaInNumGrid, moveRSigma, maxMoveDeviation, turnSigma, missMatchProbAtCoarse, coarseFactor):
        self.searchRadius = searchRadius
        self.searchHalfRad = searchHalfRad
        self.og = og
        self.scanSigmaInNumGrid = scanSigmaInNumGrid
        self.coarseFactor = coarseFactor
        self.moveRSigma = moveRSigma
        self.turnSigma = turnSigma
        self.missMatchProbAtCoarse = missMatchProbAtCoarse
        self.maxMoveDeviation = maxMoveDeviation

    def frameSearchSpace(self, estimatedX, estimatedY, unitLength, sigma, missMatchProbAtCoarse):
        maxScanRadius = 1.1 * self.og.lidarMaxRange + self.searchRadius
        xRangeList = [estimatedX - maxScanRadius, estimatedX + maxScanRadius]
        yRangeList = [estimatedY - maxScanRadius, estimatedY + maxScanRadius]
        idxEndX, idxEndY = int((xRangeList[1] - xRangeList[0]) / unitLength),  int((yRangeList[1] - yRangeList[0]) / unitLength)
        searchSpace = math.log(missMatchProbAtCoarse) * np.ones((idxEndY + 1, idxEndX + 1))

        self.og.checkAndExpandOG(xRangeList, yRangeList)
        xRangeListIdx, yRangeListIdx = self.og.convertRealXYToMapIdx(xRangeList, yRangeList)
        ogMap = self.og.occupancyGridVisited[yRangeListIdx[0]: yRangeListIdx[1], xRangeListIdx[0]: xRangeListIdx[1]] /\
                      self.og.occupancyGridTotal[yRangeListIdx[0]: yRangeListIdx[1], xRangeListIdx[0]: xRangeListIdx[1]]
        ogMap = ogMap > 0.5
        ogX = self.og.OccupancyGridX[yRangeListIdx[0]: yRangeListIdx[1], xRangeListIdx[0]: xRangeListIdx[1]]
        ogY = self.og.OccupancyGridY[yRangeListIdx[0]: yRangeListIdx[1], xRangeListIdx[0]: xRangeListIdx[1]]

        ogX, ogY = ogX[ogMap], ogY[ogMap]
        ogIdx = self.convertXYToSearchSpaceIdx(ogX, ogY, xRangeList[0], yRangeList[0], unitLength)
        searchSpace[ogIdx[1], ogIdx[0]] = 0
        probSP = self.generateProbSearchSpace(searchSpace, sigma)
        return xRangeList, yRangeList, probSP

    def generateProbSearchSpace(self, searchSpace, sigma):
        probSP = gaussian_filter(searchSpace, sigma=sigma)
        probMin = probSP.min()
        probSP[probSP > 0.5 * probMin] = 0
        return probSP

    def matchScan(self, reading, estMovingDist, estMovingTheta, count, matchMax = True):
        """Iteratively find the best dx, dy and dtheta"""
        estimatedX, estimatedY, estimatedTheta, rMeasure = reading['x'], reading['y'], reading['theta'], reading['range']
        rMeasure = np.asarray(rMeasure)
        if count == 1:
            return reading, 1
        # Coarse Search
        coarseSearchStep = self.coarseFactor * self.og.unitGridSize  # make this even number of unitGridSize for performance
        coarseSigma = self.scanSigmaInNumGrid / self.coarseFactor

        xRangeList, yRangeList, probSP = self.frameSearchSpace(estimatedX, estimatedY, coarseSearchStep, coarseSigma, self.missMatchProbAtCoarse)
        matchedPx, matchedPy, matchedReading, convTotal, coarseConfidence = self.searchToMatch(probSP, estimatedX, estimatedY,
            estimatedTheta, rMeasure, xRangeList, yRangeList, self.searchRadius,
                self.searchHalfRad, coarseSearchStep, estMovingDist, estMovingTheta,fineSearch=False, matchMax=matchMax)
        fineSearchStep = self.og.unitGridSize
        fineSigma = self.scanSigmaInNumGrid
        fineSearchHalfRad = self.searchHalfRad
        fineMissMatchProbAtFine = self.missMatchProbAtCoarse ** (2 / self.coarseFactor)
        xRangeList, yRangeList, probSP = self.frameSearchSpace(matchedReading['x'], matchedReading['y'], fineSearchStep, fineSigma, fineMissMatchProbAtFine)
        matchedPx, matchedPy, matchedReading, convTotal, fineConfidence = self.searchToMatch(probSP, matchedReading['x'],
            matchedReading['y'], matchedReading['theta'], matchedReading['range'], xRangeList, yRangeList,
                coarseSearchStep, fineSearchHalfRad, fineSearchStep, estMovingDist, estMovingTheta, fineSearch=True, matchMax=True)

        return matchedReading, coarseConfidence

    def covertMeasureToXY(self, estimatedX, estimatedY, estimatedTheta, rMeasure):
        rads = np.linspace(estimatedTheta - self.og.lidarFOV / 2, estimatedTheta + self.og.lidarFOV / 2,
                           num=self.og.numSamplesPerRev)
        range_idx = rMeasure < self.og.lidarMaxRange
        rMeasureInRange = rMeasure[range_idx]
        rads = rads[range_idx]
        px = estimatedX + np.cos(rads) * rMeasureInRange
        py = estimatedY + np.sin(rads) * rMeasureInRange
        return px, py

    def searchToMatch(self, probSP, estimatedX, estimatedY, estimatedTheta, rMeasure, xRangeList, yRangeList,
                      searchRadius, searchHalfRad, unitLength, estMovingDist,  estMovingTheta, fineSearch = False, matchMax = True):
        px, py = self.covertMeasureToXY(estimatedX, estimatedY, estimatedTheta, rMeasure)
        numCellOfSearchRadius = int(searchRadius / unitLength)
        xMovingRange = np.arange(-numCellOfSearchRadius, numCellOfSearchRadius + 1)
        yMovingRange = np.arange(-numCellOfSearchRadius, numCellOfSearchRadius + 1)
        xv, yv = np.meshgrid(xMovingRange, yMovingRange)
        if fineSearch:
            rv, thetaWeight = np.zeros(xv.shape), np.zeros(xv.shape)
        else:
            rv = - (1 / (2 * self.moveRSigma ** 2)) * (np.sqrt((xv * unitLength) ** 2 + (yv * unitLength) ** 2) - (estMovingDist)) ** 2
            rrv = np.abs(np.sqrt((xv * unitLength) ** 2 + (yv * unitLength) ** 2) - estMovingDist)
            rv[rrv > self.maxMoveDeviation] = -100
            if estMovingTheta is not None:
                distv = np.sqrt(np.square(xv) + np.square(yv))
                distv[distv == 0] = 0.0001
                thetav = np.arccos((xv * math.cos(estMovingTheta) + yv * math.sin(estMovingTheta)) / distv)
                thetaWeight = -1 / (2 * self.turnSigma ** 2) * np.square(thetav)
            else:
                thetaWeight = np.zeros(xv.shape)

        xv = xv.reshape((xv.shape[0], xv.shape[1], 1))
        yv = yv.reshape((yv.shape[0], yv.shape[1], 1))
        thetaRange = np.arange(-searchHalfRad, searchHalfRad + self.og.angularStep, self.og.angularStep)
        convTotal = np.zeros((len(thetaRange), xv.shape[0], xv.shape[1]))
        for i, theta in enumerate(thetaRange):
            rotatedPx, rotatedPy = self.rotate((estimatedX, estimatedY), (px, py), theta)
            rotatedPxIdx, rotatedPyIdx = self.convertXYToSearchSpaceIdx(rotatedPx, rotatedPy, xRangeList[0],
                                                                        yRangeList[0], unitLength)
            uniqueRotatedPxPyIdx = np.unique(np.column_stack((rotatedPxIdx, rotatedPyIdx)), axis=0)
            rotatedPxIdx, rotatedPyIdx = uniqueRotatedPxPyIdx[:, 0], uniqueRotatedPxPyIdx[:, 1]
            rotatedPxIdx = rotatedPxIdx.reshape(1, 1, -1)
            rotatedPyIdx = rotatedPyIdx.reshape(1, 1, -1)
            rotatedPxIdx = rotatedPxIdx + xv
            rotatedPyIdx = rotatedPyIdx + yv
            convResult = probSP[rotatedPyIdx, rotatedPxIdx]
            convResultSum = np.sum(convResult, axis=2)
            convResultSum = convResultSum + rv + thetaWeight
            convTotal[i, :, :] = convResultSum
        if matchMax:
            maxIdx = np.unravel_index(convTotal.argmax(), convTotal.shape)
        else:
            convTotalFlatten = np.reshape(convTotal, -1)
            convTotalFlattenProb = np.exp(convTotalFlatten) / np.exp(convTotalFlatten).sum()
            maxIdx = np.random.choice(np.arange(convTotalFlatten.size), 1, p=convTotalFlattenProb)[0]
            maxIdx = np.unravel_index(maxIdx, convTotal.shape)

        confidence = np.sum(np.exp(convTotal))
        dx, dy, dtheta = xMovingRange[maxIdx[2]] * unitLength, yMovingRange[maxIdx[1]] * unitLength, thetaRange[maxIdx[0]]
        matchedReading = {"x": estimatedX + dx, "y": estimatedY + dy, "theta": estimatedTheta + dtheta,
                          "range": rMeasure}
        matchedPx, matchedPy = self.rotate((estimatedX, estimatedY), (px, py), dtheta)

        return matchedPx + dx, matchedPy + dy, matchedReading, convTotal, confidence

    def plotMatchOverlay(self, probSP, matchedPx, matchedPy, matchedReading, xRangeList, yRangeList, unitLength):
        plt.figure(figsize=(19.20, 19.20))
        plt.imshow(probSP, origin='lower')
        pxIdx, pyIdx = self.convertXYToSearchSpaceIdx(matchedPx, matchedPy, xRangeList[0], yRangeList[0], unitLength)
        plt.scatter(pxIdx, pyIdx, c='r', s=5)
        poseXIdx, poseYIdx = self.convertXYToSearchSpaceIdx(matchedReading['x'], matchedReading['y'], xRangeList[0], yRangeList[0], unitLength)
        plt.scatter(poseXIdx, poseYIdx, color='blue', s=50)

    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.
        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point
        qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
        qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
        return qx, qy

    def convertXYToSearchSpaceIdx(self, px, py, beginX, beginY, unitLength):
        xIdx = (((px - beginX) / unitLength)).astype(int)
        yIdx = (((py - beginY) / unitLength)).astype(int)
        return xIdx, yIdx

@dataclass
class OGParameters:
    """Parameters for Occupancy Grid initialization"""
    map_x_length: float
    map_y_length: float
    init_xy: Dict
    unit_grid_size: float
    lidar_fov: float
    lidar_max_range: float
    num_samples_per_rev: int
    wall_thickness: float

@dataclass
class SMParameters:
    """Parameters for Scan Matching"""
    search_radius: float
    search_half_rad: float
    sigma_in_num_grid: float
    move_r_sigma: float
    max_move_deviation: float
    turn_sigma: float
    miss_match_prob_coarse: float
    coarse_factor: float

class ParticleFilter:
    def __init__(self, num_particles: int, og_params: OGParameters, sm_params: SMParameters):
        self.num_particles = num_particles
        self.particles = self._initialize_particles(og_params, sm_params)
        self.step = 0
        self.prev_matched_reading = None
        self.prev_raw_reading = None
        self.particles_trajectory = []

    def _initialize_particles(self, og_params: OGParameters, sm_params: SMParameters) -> List['Particle']:
        return [Particle(og_params, sm_params) for _ in range(self.num_particles)]

    def update_particles(self, reading: Dict, count: int) -> None:
        for particle in self.particles:
            particle.update(reading, count)

    def is_weight_unbalanced(self) -> bool:
        self._normalize_weights()
        variance = sum((p.weight - 1/self.num_particles)**2 for p in self.particles)
        threshold = ((self.num_particles - 1) / self.num_particles)**2 + \
                   (self.num_particles - 1.000000000000001) * (1/self.num_particles)**2
        return variance > threshold

    def _normalize_weights(self) -> None:
        weight_sum = sum(p.weight for p in self.particles)
        for particle in self.particles:
            particle.weight /= weight_sum

    def resample(self) -> None:
        weights = np.array([p.weight for p in self.particles])
        temp_particles = [copy.deepcopy(p) for p in self.particles]
        resampled_indices = np.random.choice(
            np.arange(self.num_particles), 
            self.num_particles, 
            p=weights
        )
        
        for i, idx in enumerate(resampled_indices):
            self.particles[i] = copy.deepcopy(temp_particles[idx])
            self.particles[i].weight = 1 / self.num_particles

class Particle:
    def __init__(self, og_params: OGParameters, sm_params: SMParameters):
        self.og = OccupancyGrid(
            og_params.map_x_length, og_params.map_y_length,
            og_params.init_xy, og_params.unit_grid_size,
            og_params.lidar_fov, og_params.num_samples_per_rev,
            og_params.lidar_max_range, og_params.wall_thickness
        )
        self.sm = ScanMatcher(
            self.og, sm_params.search_radius, sm_params.search_half_rad,
            sm_params.sigma_in_num_grid, sm_params.move_r_sigma,
            sm_params.max_move_deviation, sm_params.turn_sigma,
            sm_params.miss_match_prob_coarse, sm_params.coarse_factor
        )
        self.x_trajectory: List[float] = []
        self.y_trajectory: List[float] = []
        self.weight = 1.0
        self.prev_raw_moving_theta: Optional[float] = None
        self.prev_matched_moving_theta: Optional[float] = None
        self.prev_matched_reading: Optional[Dict] = None
        self.prev_raw_reading: Optional[Dict] = None

    def update_estimated_pose(self, current_raw_reading: Dict) -> Tuple[Dict, float, Optional[float], Optional[float]]:
        estimated_theta = (self.prev_matched_reading['theta'] + 
                         current_raw_reading['theta'] - 
                         self.prev_raw_reading['theta'])
        
        estimated_reading = {
            'x': self.prev_matched_reading['x'],
            'y': self.prev_matched_reading['y'],
            'theta': estimated_theta,
            'range': current_raw_reading['range']
        }

        dx = current_raw_reading['x'] - self.prev_raw_reading['x']
        dy = current_raw_reading['y'] - self.prev_raw_reading['y']
        est_moving_dist = math.sqrt(dx**2 + dy**2)
        
        raw_move = self._calculate_raw_movement(current_raw_reading)
        est_moving_theta, raw_moving_theta = self._calculate_movement_theta(
            current_raw_reading, raw_move
        )

        return estimated_reading, est_moving_dist, est_moving_theta, raw_moving_theta

    def _calculate_raw_movement(self, current_raw_reading: Dict) -> float:
        dx = current_raw_reading['x'] - self.prev_raw_reading['x']
        dy = current_raw_reading['y'] - self.prev_raw_reading['y']
        return math.sqrt(dx**2 + dy**2)

    def _calculate_movement_theta(
        self, 
        current_raw_reading: Dict, 
        raw_move: float
    ) -> Tuple[Optional[float], Optional[float]]:
        if raw_move <= 0.3:
            return None, None

        dx = current_raw_reading['x'] - self.prev_raw_reading['x']
        dy = current_raw_reading['y'] - self.prev_raw_reading['y']
        
        raw_moving_theta = math.acos(dx / raw_move) if dy > 0 else -math.acos(dx / raw_move)
        
        if self.prev_raw_moving_theta is not None:
            raw_turn_theta = raw_moving_theta - self.prev_raw_moving_theta
            est_moving_theta = self.prev_matched_moving_theta + raw_turn_theta
        else:
            est_moving_theta = None
            
        return est_moving_theta, raw_moving_theta

    def get_moving_theta(self, matched_reading: Dict) -> Optional[float]:
        if not self.x_trajectory:
            return None
            
        dx = matched_reading['x'] - self.x_trajectory[-1]
        dy = matched_reading['y'] - self.y_trajectory[-1]
        move = math.sqrt(dx**2 + dy**2)
        
        if move == 0:
            return None
            
        return math.acos(dx / move) if dy > 0 else -math.acos(dx / move)

    def update(self, reading: Dict, count: int) -> None:
        if count == 1:
            self.prev_raw_moving_theta = None
            self.prev_matched_moving_theta = None
            matched_reading, confidence = reading, 1
        else:
            estimated_reading, est_moving_dist, est_moving_theta, raw_moving_theta = \
                self.update_estimated_pose(reading)
            
            matched_reading, confidence = self.sm.matchScan(
                estimated_reading, est_moving_dist, est_moving_theta, count, matchMax=False
            )
            
            self.prev_raw_moving_theta = raw_moving_theta
            self.prev_matched_moving_theta = self.get_moving_theta(matched_reading)

        self._update_trajectory(matched_reading)
        self.og.updateOccupancyGrid(matched_reading)
        self.prev_matched_reading = matched_reading
        self.prev_raw_reading = reading
        self.weight *= confidence

    def _update_trajectory(self, matched_reading: Dict) -> None:
        self.x_trajectory.append(matched_reading['x'])
        self.y_trajectory.append(matched_reading['y'])

    def _plot_trajectory(self) -> None:
        plt.scatter(self.x_trajectory[0], self.y_trajectory[0], color='r', s=500)
        
        colors = iter(cm.rainbow(np.linspace(1, 0, len(self.x_trajectory) + 1)))
        for x, y in zip(self.x_trajectory, self.y_trajectory):
            plt.scatter(x, y, color=next(colors), s=35)
            
        plt.scatter(self.x_trajectory[-1], self.y_trajectory[-1], 
                   color=next(colors), s=500)
        plt.plot(self.x_trajectory, self.y_trajectory)

class SLAMProcessor:
    def __init__(self, particle_filter: ParticleFilter):
        self.pf = particle_filter
        self.output_dir = Path('Output')
        self.output_dir.mkdir(exist_ok=True)

    def process_sensor_data(self, sensor_data: Dict, plot_trajectory: bool = True) -> None:
        for count, key in enumerate(sorted(sensor_data.keys()), 1):
            print(f"Processing step {count}")
            self._process_single_reading(sensor_data[key], count)
            self._save_plot(count)

    def _process_single_reading(self, reading: Dict, count: int) -> None:
        self.pf.update_particles(reading, count)
        if self.pf.is_weight_unbalanced():
            print("Resampling particles...")
            self.pf.resample()

    def _save_plot(self, count: int) -> None:
        plt.figure(figsize=(19.20, 19.20))
        best_particle = self._plot_current_state()
        
        output_file = self.output_dir / f'{str(count).zfill(3)}.png'
        plt.savefig(output_file)
        plt.close()

    def _plot_current_state(self) -> Particle:
        best_particle = max(self.pf.particles, key=lambda p: p.weight)
        
        for particle in self.pf.particles:
            plt.plot(particle.x_trajectory, particle.y_trajectory)
            
        self._plot_occupancy_grid(best_particle)
        return best_particle

    def _plot_occupancy_grid(self, particle: Particle) -> None:
        x_range, y_range = [-13, 20], [-25, 7]
        og_map = particle.og.occupancyGridVisited / particle.og.occupancyGridTotal
        x_idx, y_idx = particle.og.convertRealXYToMapIdx(x_range, y_range)
        og_map = og_map[y_idx[0]: y_idx[1], x_idx[0]: x_idx[1]]
        og_map = np.flipud(1 - og_map)
        plt.imshow(og_map, cmap='gray', extent=[x_range[0], x_range[1], 
                                               y_range[0], y_range[1]])

def read_json(json_file: str) -> Dict:
    with open(json_file, 'r') as f:
        return json.load(f)['map']

def main():
    og_params = OGParameters(
        map_x_length=50,
        map_y_length=50,
        unit_grid_size=0.02,
        lidar_fov=np.pi,
        lidar_max_range=10,
        wall_thickness=0.1,  # 5 * unit_grid_size
        num_samples_per_rev=0,  # Will be set from sensor data
        init_xy=None  # Will be set from sensor data
    )

    sm_params = SMParameters(
        search_radius=1.4,
        search_half_rad=0.25,
        sigma_in_num_grid=2,
        move_r_sigma=0.1,
        max_move_deviation=0.25,
        turn_sigma=0.3,
        miss_match_prob_coarse=0.15,
        coarse_factor=5
    )

    sensor_data = read_json("DataSet/intel.json")
    first_reading = sensor_data[sorted(sensor_data.keys())[0]]
    
    og_params.num_samples_per_rev = len(first_reading['range'])
    og_params.init_xy = first_reading

    pf = ParticleFilter(num_particles=10, og_params=og_params, sm_params=sm_params)
    processor = SLAMProcessor(pf)
    processor.process_sensor_data(sensor_data, plot_trajectory=True)

    create_gif_from_images('Output', 'output.gif', duration=100)
    print("GIF created successfully!")

    latitude = 45.5222
    longitude = -122.8234
    top_left = (longitude, latitude)  

    convert_to_geotiff(png_path='Output/910.png', geotiff_path='result.tif', top_left=top_left, pixel_size=0.0001)

if __name__ == '__main__':
    main()