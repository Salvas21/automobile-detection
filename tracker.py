import math


class EuclideanDistTracker:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}

        # Keep the count of the IDs, each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rectangles):
        # Objects bounding boxes and ids
        objects_bb_ids = []

        # Get center point of new object
        for rectangle in objects_rectangles:
            x, y, width, height, index = rectangle
            cx = (x + x + width) // 2
            cy = (y + y + height) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, point in self.center_points.items():
                distance = math.hypot(cx - point[0], cy - point[1])

                # can change the detection distance threshold (basic = 25)
                if distance < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bb_ids.append([x, y, width, height, id, index])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bb_ids.append([x, y, width, height, self.id_count, index])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for object_bb_id in objects_bb_ids:
            _, _, _, _, object_id, index = object_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary
        self.center_points = new_center_points.copy()
        return objects_bb_ids
