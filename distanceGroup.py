from collections import defaultdict

class DistanceGroup():
    
    def __init__(self, range_value):
        # thumb
        self.thumbWidthRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.thumbWidthRange.insert(0, (0, 0.001))
        self.thumbWidthRange_containers = defaultdict(list)
        self.thumbHeightRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.thumbHeightRange.insert(0, (0, 0.001))
        self.thumbHeightRange_containers = defaultdict(list)

        # index finger
        self.indexWidthRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.indexWidthRange.insert(0, (0, 0.001))
        self.indexWidthRange_containers = defaultdict(list)
        self.indexHeightRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.indexHeightRange.insert(0, (0, 0.001))
        self.indexHeightRange_containers = defaultdict(list)

        # middle finger
        self.middleWidthRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.middleWidthRange.insert(0, (0, 0.001))
        self.middleWidthRange_containers = defaultdict(list)
        self.middleHeightRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.middleHeightRange.insert(0, (0, 0.001))
        self.middleHeightRange_containers = defaultdict(list)

        # ring finger
        self.ringWidthRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.ringWidthRange.insert(0, (0, 0.001))
        self.ringWidthRange_containers = defaultdict(list)
        self.ringHeightRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.ringHeightRange.insert(0, (0, 0.001))
        self.ringHeightRange_containers = defaultdict(list)

        # pinky finger
        self.pinkyWidthRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.pinkyWidthRange.insert(0, (0, 0.001))
        self.pinkyWidthRange_containers = defaultdict(list)
        self.pinkyHeightRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.pinkyHeightRange.insert(0, (0, 0.001))
        self.pinkyHeightRange_containers = defaultdict(list)
        
        # thumb to index finger
        self.thumbToIndexTipWidthRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.thumbToIndexTipWidthRange.insert(0, (0, 0.001))
        self.thumbToIndexTipWidthRange_containers = defaultdict(list)
        self.thumbToIndexTipHeightRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.thumbToIndexTipHeightRange.insert(0, (0, 0.001))
        self.thumbToIndexTipHeightRange_containers = defaultdict(list)
        
        # index to middle finger
        self.indexToMiddleTipWidthRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.indexToMiddleTipWidthRange.insert(0, (0, 0.001))
        self.indexToMiddleTipWidthRange_containers = defaultdict(list)
        self.indexToMiddleTipHeightRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.indexToMiddleTipHeightRange.insert(0, (0, 0.001))
        self.indexToMiddleTipHeightRange_containers = defaultdict(list)
        
        # middle to ring finger
        self.middleToRingTipWidthRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.middleToRingTipWidthRange.insert(0, (0, 0.001))
        self.middleToRingTipWidthRange_containers = defaultdict(list)
        self.middleToRingTipHeightRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.middleToRingTipHeightRange.insert(0, (0, 0.001))
        self.middleToRingTipHeightRange_containers = defaultdict(list)
        
        # ring to pinky finger
        self.ringToPinkyTipWidthRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.ringToPinkyTipWidthRange.insert(0, (0, 0.001))
        self.ringToPinkyTipWidthRange_containers = defaultdict(list)
        self.ringToPinkyTipHeightRange = [(i/range_value, (i+1)/range_value) for i in range(1, range_value)]
        self.ringToPinkyTipHeightRange.insert(0, (0, 0.001))
        self.ringToPinkyTipHeightRange_containers = defaultdict(list)


    
    # IMPORTANT: ONLY ONE GESTURE call this function at any point of time.
    # Once close or any gesture detected, call this function to collect all x and y distances of each finger 
    def getLandMarkWidthAndHeightDistanceOfOneGestureAllFingers(self, landmarks):
        
        thumb_tip = landmarks[4]
        thumb_base = landmarks[1]
        index_tip = landmarks[8]
        index_middle = landmarks[6]
        middle_tip = landmarks[12]
        middle_middle = landmarks[10]
        ring_tip = landmarks[16]
        ring_middle = landmarks[14]
        pinky_tip = landmarks[20]
        pinky_middle = landmarks[18]
        base = landmarks[0]
        
        # thumb x and y distance grouping of this one gesture
        self.getLandmarkWidthDistanceGroup(thumb_tip, thumb_base, self.thumbWidthRange, self.thumbWidthRange_containers)
        self.getLandmarkHeightDistanceGroup(thumb_tip, thumb_base, self.thumbHeightRange, self.thumbHeightRange_containers)
        
        # index finger x and y distance grouping of this one gesture
        self.getLandmarkWidthDistanceGroup(index_tip, index_middle, self.indexWidthRange, self.indexWidthRange_containers)
        self.getLandmarkHeightDistanceGroup(index_tip, index_middle, self.indexHeightRange, self.indexHeightRange_containers)
        
        # middle x and y distance grouping of this one gesture
        self.getLandmarkWidthDistanceGroup(middle_tip, middle_middle, self.middleWidthRange, self.middleWidthRange_containers)
        self.getLandmarkHeightDistanceGroup(middle_tip, middle_middle, self.middleHeightRange, self.middleHeightRange_containers)
        
        # ring x and y distance grouping of this one gesture
        self.getLandmarkWidthDistanceGroup(ring_tip, ring_middle, self.ringWidthRange, self.ringWidthRange_containers)
        self.getLandmarkHeightDistanceGroup(ring_tip, ring_middle, self.ringHeightRange, self.ringHeightRange_containers)
        
        # pinky x and y distance grouping of this one gesture
        self.getLandmarkWidthDistanceGroup(pinky_tip, pinky_middle, self.pinkyWidthRange, self.pinkyWidthRange_containers)
        self.getLandmarkHeightDistanceGroup(pinky_tip, pinky_middle, self.pinkyHeightRange, self.pinkyHeightRange_containers)
        
        # thumb to index finger tip x and y distance grouping of this one gesture
        self.getLandmarkWidthDistanceGroup(thumb_tip, index_tip, self.thumbToIndexTipWidthRange, self.thumbToIndexTipWidthRange_containers)
        self.getLandmarkHeightDistanceGroup(thumb_tip, index_tip, self.thumbToIndexTipHeightRange, self.thumbToIndexTipHeightRange_containers)
        
        # middle x and y distance grouping of this one gesture
        self.getLandmarkWidthDistanceGroup(index_tip, middle_tip, self.indexToMiddleTipWidthRange, self.indexToMiddleTipWidthRange_containers)
        self.getLandmarkHeightDistanceGroup(index_tip, middle_tip, self.indexToMiddleTipHeightRange, self.indexToMiddleTipHeightRange_containers)
        
        # ring x and y distance grouping of this one gesture
        self.getLandmarkWidthDistanceGroup(middle_tip, ring_tip, self.middleToRingTipWidthRange, self.middleToRingTipWidthRange_containers)
        self.getLandmarkHeightDistanceGroup(middle_tip, ring_tip, self.middleToRingTipHeightRange, self.middleToRingTipHeightRange_containers)
        
        # pinky x and y distance grouping of this one gesture
        self.getLandmarkWidthDistanceGroup(ring_tip, pinky_tip, self.ringToPinkyTipWidthRange, self.ringToPinkyTipWidthRange_containers)
        self.getLandmarkHeightDistanceGroup(ring_tip, pinky_tip, self.ringToPinkyTipHeightRange, self.ringToPinkyTipHeightRange_containers)
        



    # once close gesture detected, call this function to check width distance and grouping for each finger
    def getLandmarkWidthDistanceGroup(self, landmark1, landmark2, widthRanges, widthRange_container):
        
        width_distance = abs(landmark1.x - landmark2.x)
        
        self.categorisePoint(width_distance, widthRanges, widthRange_container)




    # once close gesture detected, call this function to check height distance and grouping for each finger
    def getLandmarkHeightDistanceGroup(self, landmark1, landmark2, heightRanges, heightRange_container):
        
        height_distance = abs(landmark1.y - landmark2.y)
        
        self.categorisePoint(height_distance, heightRanges, heightRange_container)    




    # x or y grouping based on distances between two part of each finger 
    def categorisePoint(self, distance, ranges, range_containers):
        for lower, upper in ranges:
            if lower <= distance < upper:
                range_containers[(lower, upper)].append(distance)
                break




    def printAndRemoveMaximumDistanceGroup(self, range_container):
        total = len(range_container)
        maxRange, maxCount = self.getMaximumDistanceGroup(range_container)
        print("1st champion range : "+str(maxRange) + ", [ " + str(maxCount) + "/"+ str(total) + " ]")
        
        if maxRange is not None:
            del range_container[maxRange]
        
        secondMaxRange, secondMaxCount = self.getMaximumDistanceGroup(range_container)
        print("2nd max range : "+str(secondMaxRange) + ", [ " + str(secondMaxCount) + "/"+ str(total) + " ]")
    
    
    
        
    def getMaximumDistanceGroup(self, range_container):
        maxRange = None
        maxCount = 0
        
        for key, value in range_container.items():
            if len(value) > maxCount:
                maxCount = len(value)
                maxRange = key
        
        return maxRange, maxCount



    def printHighestDistanceGroupingStatistic(self):
        
        print("1st and 2nd max group for thumb width")
        self.printAndRemoveMaximumDistanceGroup(self.thumbWidthRange_containers)
        print("1st and 2nd max group for thumb height")
        self.printAndRemoveMaximumDistanceGroup(self.thumbHeightRange_containers)
                        
        print("1st and 2nd max group for index width")
        self.printAndRemoveMaximumDistanceGroup(self.indexWidthRange_containers)
        print("1st and 2nd max group for thumb height")
        self.printAndRemoveMaximumDistanceGroup(self.indexHeightRange_containers)
                        
        print("1st and 2nd max group for middle width")
        self.printAndRemoveMaximumDistanceGroup(self.middleWidthRange_containers)
        print("1st and 2nd max group for middle height")
        self.printAndRemoveMaximumDistanceGroup(self.middleHeightRange_containers)
                        
        print("1st and 2nd max group for ring width")
        self.printAndRemoveMaximumDistanceGroup(self.ringWidthRange_containers)
        print("1st and 2nd max group for ring height")
        self.printAndRemoveMaximumDistanceGroup(self.ringWidthRange_containers)
                        
        print("1st and 2nd max group for pinky width")
        self.printAndRemoveMaximumDistanceGroup(self.pinkyWidthRange_containers)
        print("1st and 2nd max group for pinky height")
        self.printAndRemoveMaximumDistanceGroup(self.pinkyHeightRange_containers)
            
        print("1st and 2nd max group for thumb to index tip width")
        self.printAndRemoveMaximumDistanceGroup(self.thumbToIndexTipWidthRange_containers)
        print("1st and 2nd max group for thumb to index tip height")
        self.printAndRemoveMaximumDistanceGroup(self.thumbToIndexTipHeightRange_containers)
                        
        print("1st and 2nd max group for index to middle tip width")
        self.printAndRemoveMaximumDistanceGroup(self.indexToMiddleTipWidthRange_containers)
        print("1st and 2nd max group for index to middle tip height")
        self.printAndRemoveMaximumDistanceGroup(self.indexToMiddleTipHeightRange_containers)
                        
        print("1st and 2nd max group for middle to ring tip width")
        self.printAndRemoveMaximumDistanceGroup(self.middleToRingTipWidthRange_containers)
        print("1st and 2nd max group for middle to ring tip height")
        self.printAndRemoveMaximumDistanceGroup(self.middleToRingTipHeightRange_containers)
                        
        print("1st and 2nd max group for ring to pinky tip width")
        self.printAndRemoveMaximumDistanceGroup(self.ringToPinkyTipWidthRange_containers)
        print("1st and 2nd max group for ring top pinky tip height")
        self.printAndRemoveMaximumDistanceGroup(self.ringToPinkyTipHeightRange_containers)