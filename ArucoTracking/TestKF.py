# Example Kalman Filter using this example:
# https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
# Testing out FilterPy package.

# Imports
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import numpy as np
import matplotlib.pyplot as plt

f = KalmanFilter(dim_x=3, dim_z=3)  # set up Kalman Filter
dt = 0.100  # sample rate

# Initial state
# For this example, let's use position [m] and velocity [m/s] and acceleration [m/s2]
# So, x = [position, velocity, acceleration]
f.x = np.array([0., 1., 0.])

# State transition matrix
# Used for predictive model: x_predicted = F * x_prev
f.F = np.array([[1., dt, 0.5*dt*dt],
                [0., 1., dt],
                [0., 0., 1.]])

# Measurement matrix
# Maps measurements and states: measurement = H * state
# In this case, let's say we can read both position and velocity
f.H = np.array([[1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 1.]])

# Covariance matrix
# Measure of uncertainty in state estimate - this is updated every loop
f.P *= 100.

# Measurement noise
sensorVariance = 0.2
f.R *= sensorVariance

# Process noise
# Uncertainty in our predictive model
f.Q = Q_discrete_white_noise(dim=3, dt=0.1, var=0.13)

# In our actual Kalman filter, we'll use real-time data here
# In this case, let's say we're measuring the position of an object moving in a sinusoidal pattern
# and we'll add some Gaussian noise too!
sampleTimes = np.arange(0, 10, dt)
measPosition = np.sin(sampleTimes) + np.random.normal(0.0, np.sqrt(sensorVariance), len(sampleTimes))
measVelocity = np.cos(sampleTimes) + np.random.normal(0.0, np.sqrt(sensorVariance), len(sampleTimes))
measAccel = -1 * np.sin(sampleTimes) + np.random.normal(0.0, np.sqrt(sensorVariance), len(sampleTimes))

# Now we actually use the Kalman Filter.
filteredPosition = np.zeros(len(sampleTimes))
filteredVelocity = np.zeros(len(sampleTimes))
filteredAccel = np.zeros(len(sampleTimes))
for i in range(0, len(measPosition)):
    z = np.array([measPosition[i], measVelocity[i], measAccel[i]]).T
    f.predict()
    f.update(z)
    filteredPosition[i] = f.x[0]
    filteredVelocity[i] = f.x[1]
    filteredAccel[i] = f.x[2]

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
ax1.plot(sampleTimes, measPosition, label = "Measured Position")
ax1.plot(sampleTimes, filteredPosition, label = "Filtered Position")
ax1.legend()
ax2.plot(sampleTimes, measVelocity, label = "Measured Velocity")
ax2.plot(sampleTimes, filteredVelocity, label = "Filtered Velocity")
ax2.legend()
ax3.plot(sampleTimes, measAccel, label = "Measured Acceleration")
ax3.plot(sampleTimes, filteredAccel, label = "Filtered Acceleration")
ax3.legend()
plt.show()
