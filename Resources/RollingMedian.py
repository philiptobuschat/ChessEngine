import heapq

class RollingMedian:
    '''
    efficient calculation of rolling median
    '''
    def __init__(self):
        self.smaller_half = []  # max heap (values negated for max heap behavior)
        self.larger_half = []   # min heap

    def add_number(self, num):
        if len(self.smaller_half) == 0 or num <= -self.smaller_half[0]:
            # smaller half empty or new number smaller than largest of small half 
            # -> add new number to smaller half (negated as all numbers in smaller heap)
            heapq.heappush(self.smaller_half, -num)
        else:
            # otherwise -> add new number to larger half
            heapq.heappush(self.larger_half, num)

        # ensure the heaps are balanced
        if len(self.smaller_half) > len(self.larger_half) + 1:
            heapq.heappush(self.larger_half, -heapq.heappop(self.smaller_half))
        elif len(self.larger_half) > len(self.smaller_half):
            heapq.heappush(self.smaller_half, -heapq.heappop(self.larger_half))

    def get_median(self):
        # heaps either have same number of elements or smaller half has one element more
        if len(self.smaller_half) == len(self.larger_half):
            if len(self.smaller_half) == 0:
                return None
            return (-self.smaller_half[0] + self.larger_half[0]) / 2.0
        else:
            return -self.smaller_half[0]