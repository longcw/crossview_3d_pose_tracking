import random


class ColorGenerator:
    def __init__(self):
        self.points = []
        self.colors = []
        self.id_to_color = {0: [0, 0, 0]}

    def get_new_point(self):
        points = self.points
        if len(points) == 0:
            points.append(0)
            return 0
        elif len(points) == 1:
            points.append(255)
            return 255
        elif len(points) == 256:
            return False

        max_distance = 0
        max_idx = None
        for i in range(len(points) - 1):
            distance = points[i + 1] - points[i]
            if distance > max_distance:
                max_distance = distance
                max_idx = i

        new_point = points[max_idx] + max_distance // 2
        # new_point = random.randint(self.points[max_idx] + 1 + max_distance//4,
        #         self.points[max_idx+1] - 1 - max_distance//4)
        points.insert(max_idx + 1, new_point)

        return new_point

    def generate_colors(self):
        def is_allowed(color):
            return any(x > 30 for x in color) and any(x < 225 for x in color)

        colors = set()
        new_point = self.get_new_point()
        points = self.points
        for point_i in points:
            for point_j in points:
                colors.add((new_point, point_i, point_j))
                colors.add((point_i, new_point, point_j))
                colors.add((point_i, point_j, new_point))
        colors = [color for color in colors if is_allowed(color)]
        random.Random(0).shuffle(colors)

        self.colors = colors

    def get_color(self, ind):
        if ind in self.id_to_color:
            return self.id_to_color[ind]
        while len(self.colors) == 0:
            self.generate_colors()
        color = self.colors.pop()
        self.id_to_color[ind] = color
        return color
