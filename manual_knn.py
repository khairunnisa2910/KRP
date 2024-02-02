class KNearestNeighbors:
    def __init__(self, k):
        self.k = k
        self.data = []

    def fit(self, X, y):
        self.data = [(x, label) for x, label in zip(X, y)]

    def euclidean_distance(self, point1, point2):
        return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

    def find_neighbors(self, point):
        distances = sorted((self.euclidean_distance(point, x), label) for x, label in self.data)
        return distances[:self.k]

    def predict(self, point):
        neighbors = self.find_neighbors(point)
        labels = [label for _, label in neighbors]
        return max(set(labels), key=labels.count)

    def detailed_predict(self, point, id):
        # Menghitung jarak dan menyimpan detail perhitungan
        distances = []
        for i, (x, label) in enumerate(self.data):
            distance = self.euclidean_distance(point, x)
            distance_details = {
                'id': id[i], 
                'point1': point,
                'point2': x,
                'distance': distance,
                'components': [(x[j] - point[j]) ** 2 for j in range(len(point))]
            }
            distances.append((distance_details, label))

        sorted_distances = sorted(distances, key=lambda x: x[0]['distance'])
        k_nearest = sorted_distances[:self.k]

        # Menyimpan detail
        detail = {
            'point': point,
            'k': self.k,
            'distances': sorted_distances,
            'k_nearest': k_nearest,
            'vote_result': max(set(label for _, label in k_nearest), key=lambda l: sum(1 for _, label2 in k_nearest if label2 == l))
        }

        return detail
