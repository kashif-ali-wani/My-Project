import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define symbols
phi, m = sp.symbols('phi m')
V = (1/2) * m**2 * phi**2  # Chaotic inflation potential

# First and second derivatives of the potential with respect to phi
V_prime = sp.diff(V, phi)
V_double_prime = sp.diff(V_prime, phi)

# Define the slow-roll parameters epsilon and eta
epsilon = (1 / 2) * (V_prime / V)**2
eta = V_double_prime / V

# Print the symbolic expressions
print("Potential V(φ):",V)
print("\nFirst derivative V'(φ):",V_prime)
print("\nSecond derivative V''(φ):",V_double_prime)
print("\nSlow-roll parameter ε(φ):",epsilon)
print("\nSlow-roll parameter η(φ):",eta)


# Define the slow-roll parameters n, alpha and r
n = 1 - 6 * epsilon + 2 * eta
print("\nValue of n is:",n)

alpha = 16 * epsilon * eta - 24 * epsilon**2
print("\nValue of α is:",alpha)

r = 16 * epsilon
print("\nValue of r is:",r)

# Lambda functions for numerical evaluation
V_func = sp.lambdify([phi, m], V, 'numpy')
V_prime_func = sp.lambdify([phi, m], V_prime, 'numpy')
V_double_prime_func = sp.lambdify([phi, m], V_double_prime, 'numpy')
epsilon_func = sp.lambdify([phi, m], epsilon, 'numpy')
eta_func = sp.lambdify([phi, m], eta, 'numpy')

# Set a value for m (mass of the inflaton) and phi range
m_val = 1e-6  # Example value for m
phi_vals = np.linspace(0.1, 15, 500)  # Avoid zero to prevent division by zero

# Compute potential and parameters
V_vals = V_func(phi_vals, m_val)
epsilon_vals = epsilon_func(phi_vals, m_val)
eta_vals = eta_func(phi_vals, m_val)

# Plot the potential V(φ)
plt.figure(figsize=(10, 6))
plt.plot(phi_vals, V_vals, label=r'$V(\phi) = \frac{1}{2} m^2 \phi^2$', color='blue')
plt.title('Chaotic Inflation Potential $V(\\phi)$', fontsize=16)
plt.xlabel(r'$\phi$', fontsize=14)
plt.ylabel(r'$V(\phi)$', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()

# Plot the slow-roll parameter ε
plt.figure(figsize=(10, 6))
plt.plot(phi_vals, epsilon_vals, label=r'$\epsilon(\phi)$', color='red')
plt.title('Slow-roll Parameters $\\epsilon(\\phi)$', fontsize=16)
plt.xlabel(r'$\phi$', fontsize=14)
plt.ylabel(r'$\epsilon(\phi)$', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()

# Plot the slow-roll parameter η
plt.figure(figsize=(10,6))
plt.plot(phi_vals, eta_vals, label=r'$\eta(\phi)$', color='green')
plt.title('Slow-roll parameter $\\eta(\\phi)$', fontsize=16)
plt.xlabel(r'$\phi$', fontsize=14)
plt.ylabel(r'$\eta(\phi)$', fontsize=14)
plt.grid(True)
plt.legend()
plt.show()