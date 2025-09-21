import math

class CalculationAmplitudeClass:
    def create_vector(self, coord1, coord2):
        # Função que cria um vetor a partir de duas coordenadas
        # v = x_v2 - x_v1, y_v2 - y_v1
        return (coord2.x - coord1.x, coord2.y - coord1.y)

    def modulation_vector(self, vector):
        # Calcula a modulação (magnitude) do vetor
        # |v| = sqrt(x^2 + y^2)
        return (vector[0]**2 + vector[1]**2) ** 0.5

    def convert_degrees(self, radians):
        # Converte radianos para graus
        return radians * 180 / 3.14159265

    def calculate_amplitude(self, vector_1, vector_2):
        # Calcula o ângulo entre dois vetores usando o produto escalar
        # cos(θ) = (v1 . v2) / (|v1| * |v2|)
        scalar = ((vector_1[0] * vector_2[0]) + (vector_1[1] * vector_2[1]))
        result_modulus = self.modulation_vector(vector_1) * self.modulation_vector(vector_2)
        cosine = scalar / result_modulus
        return self.convert_degrees(math.acos(cosine))