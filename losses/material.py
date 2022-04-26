class Material():
    '''
    This class stores parameters for the StVK material model
    '''

    def __init__(self, density,       # Fabric density (kg / m2)
                       thickness,     # Fabric thickness (m)
                       young_modulus, 
                       poisson_ratio,
                       bending_multiplier=1.0,
                       stretch_multiplier=1.0):
                       
        self.density = density
        self.thickness = thickness
        self.young_modulus = young_modulus
        self.poisson_ratio = poisson_ratio

        self.bending_multiplier = bending_multiplier
        self.stretch_multiplier = stretch_multiplier

        # Bending and stretching coefficients (ARCSim)
        self.A = young_modulus / (1.0 - poisson_ratio**2)
        self.stretch_coeff = self.A
        self.stretch_coeff *= stretch_multiplier

        self.bending_coeff = self.A / 12.0 * (thickness ** 3) 
        self.bending_coeff *= bending_multiplier

        # Lamé coefficients
        self.lame_mu =  0.5 * self.stretch_coeff * (1.0 - self.poisson_ratio)
        self.lame_lambda = self.stretch_coeff * self.poisson_ratio