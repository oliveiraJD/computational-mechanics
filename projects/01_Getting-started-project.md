---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 2 hours later the temperature is 74$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
Temp = {11:85, 13:74} #Creates a dictionary with time as key and temperature at time value
T_am = 65
dT = (T[11+2]-T[11])/2 #Calculate rate of temperature change using FDA
k = -dT/(T[11]-T_am)
print(k)
```

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
def k_constant(Ti,To,Tam,time):
    dT = (To-Ti)/time
    return -dT/(Ti-Tam)
```

```{code-cell} ipython3
k_constant(85,74,65,2)
```

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
#Analytical solution
k = 0.275
T_0 = 85
Tam = 65

def TOD_an(T_0,Tam, k, time):
    '''Analytical solution for the temperature of a corpse given time since discovery
    
        Arguments 
    ---------
    T_0: temperature of the corpse at time of discovery i.e. t = 0 hours
    Tam: constant ambient temperature, degF
    k: empirical constant, dimensionless
    time: elapsed time since discovery, hours
    
    Returns
    -------
    T: temperature of the corpse at elapsed time, hours'''
    
    T = Tam + (T_0 - Tam)*np.exp(-k*time)
    return T

print(TOD_an(T_0,Tam,k,0))
```

```{code-cell} ipython3
#Numerical solution. Integration of Newton's law of cooling dT/dt
def TOD_num(T_0,Tam, k, time, step):
    t = np.linspace(0, time, step)
    dt = t[1]-t[0]
    
    T = np.zeros(len(t))
    T[0] = T_0
    
    for i in range (1, len(t)):
        T[i] = T[i-1] - k*(T[i-1] - Tam)*dt
    return T

TOD_num(T_0,Tam,.275,-5,50)
```

```{code-cell} ipython3
degree_sign = u"\N{DEGREE SIGN}"

time = 24
step_size = 12
t = np.linspace(0, time, step_size)

fig1, ax1 = plt.subplots()
ax1.plot(t,TOD_an(T_0, Tam, k, t),'-',label='analytical')
ax1.plot(t,TOD_num(T_0,Tam,k,time,step_size),'o-',label='numerical')
ax1.set_title('Step Size = ' + str(step_size))
ax1.legend()
ax1.set_xlabel('Time After Discovery (hours)')
ax1.set_ylabel('Temperature of Corpse (' + degree_sign +'F)')

time = 24
step_size = 25
t = np.linspace(0, time, step_size)

fig1, ax1 = plt.subplots()
ax1.plot(t,TOD_an(T_0, Tam, k, t),'-',label='analytical')
ax1.plot(t,TOD_num(T_0,Tam,k,time,step_size),'o-',label='numerical')
ax1.set_title('Step Size = ' + str(step_size))
ax1.legend()
ax1.set_xlabel('Time After Discovery (hours)')
ax1.set_ylabel('Temperature of Corpse (' + degree_sign +'F)')

time = 24
step_size = 50
t = np.linspace(0, time, step_size)

fig1, ax1 = plt.subplots()
ax1.plot(t,TOD_an(T_0, Tam, k, t),'-',label='analytical')
ax1.plot(t,TOD_num(T_0,Tam,k,time,step_size),'o-',label='numerical')
ax1.set_title('Step Size = ' + str(step_size))
ax1.legend()
ax1.set_xlabel('Time After Discovery (hours)')
ax1.set_ylabel('Temperature of Corpse (' + degree_sign +'F)')
```

```{code-cell} ipython3
#1b
print("Temperature of the corpse approaches ambient temperature at t approaches infinity")
```

```{code-cell} ipython3
#1c
print("The corpse was alive 1.9 hours before 11 am, so 9:06 am is time of death")
TOD_num(T_0,Tam,.275,-1.9,50)
```

4. Now that we have a working numerical model, we can look at the results if the
ambient temperature is not constant i.e. T_a=f(t). We can use the weather to improve our estimate for time of death. Consider the following Temperature for the day in question. 

    |time| Temp ($^o$F)|
    |---|---|
    |6am|50|
    |7am|51|
    |8am|55|
    |9am|60|
    |10am|65|
    |11am|70|
    |noon|75|
    |1pm|80|

    a. Create a function that returns the current temperature based upon the time (0 hours=11am, 65$^{o}$F) 
    *Plot the function $T_a$ vs time. Does it look correct? Is there a better way to get $T_a(t)$?

    b. Modify the Euler approximation solution to account for changes in temperature at each hour. 
    Compare the new nonlinear Euler approximation to the linear analytical model. 
    At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
time_vals =np.array([-5,-4,-3,-2,-1,0,1,2])
Temp_vals =np.array([50,51,55,60,65,70,75,80])

def ambient_temp(time):
    if time <= time_vals[0]:
        return Temp_vals[0]
    elif time > time_vals[0] and time <= time_vals[1]:
        return Temp_vals[1]
    elif time > time_vals[1] and time <= time_vals[2]:
        return Temp_vals[2]
    elif time > time_vals[2] and time <= time_vals[3]:
        return Temp_vals[3]
    elif time > time_vals[3] and time <= time_vals[4]:
        return Temp_vals[4]
    elif time > time_vals[4] and time <= time_vals[5]:
        return Temp_vals[5]
    elif time > time_vals[5] and time <= time_vals[6]:
        return Temp_vals[6]
    else:
        return 80
print(ambient_temp(0))  
def ambient_array(t_array):
    T_amb = np.zeroes(len(t_array))
    T_amb[np.logical_and(t_array > -5, t_array <= -4)]
    return T_amb
```

```{code-cell} ipython3
#Returns an array of temperatures given an array of time
time = np.linspace(-6,3,10)
T_amb = np.array([ambient_temp(time[i]) 
                  for i in range(0, len(time))])
print(time)
print(T_amb)

#Plotting T-amb function over time
fig2, ax2 = plt.subplots()
ax2.plot(time, T_amb,'o-',label='numerical')
ax2.set_title('Ambient Temperature over Time')
ax2.set_xlabel('Time, 0 = 11am (hours)')
ax2.set_ylabel('Temperature (' + degree_sign +'F)')
```

```{code-cell} ipython3
#Numerical solution modified with T-amb function
t = np.linspace(0,-4,24)
dt = t[1] - t[0]
T_num = np.zeros(len(t))
T_num[0] = 85
k = .275
for i in range (1, len(t)):
    T_num[i] = T_num[i-1] - k*(T_num[i-1] - ambient_temp(t[i-1]))*dt

t1 = np.linspace(0,12,24)
dt1 = t1[1] - t1[0]
T_num1 = np.zeros(len(t1))
T_num1[0] = 85
k = .275
for i in range (1, len(t1)):
    T_num1[i] = T_num1[i-1] - k*(T_num1[i-1] - ambient_temp(t[i-1]))*dt1

t_an = np.linspace(-4,12,24)
fig3, ax3 = plt.subplots()
ax3.plot(t_an,TOD_an(T_0, 70, k, t_an),'-',color='red', label='analytical')
ax3.plot(t,T_num,'o-',color= 'blue', label='numerical')
ax3.plot(t1,T_num1, 'o-', color='blue')
ax3.legend()
ax3.set_xlabel('Time 0=11am (hours)')
ax3.set_ylabel('Temperature of Corpse (' + degree_sign +'F)')
```

```{code-cell} ipython3
#4b
"We see that the analytical solution converges to the constant ambient temperature of 70 after a couple of hours"
"The numerical solution continues to decrease because the ambient temperature continues to change with time"
```

```{code-cell} ipython3
import math
x=0
for i in range (0,len(T_num)):
    if math.isclose(T_num[i], 98.6, abs_tol = 1):
        x=i
print(x)
print(T_num[x])
print(T_num)
print(dt*x)
```

```{code-cell} ipython3
print("The nonlinear Euler approximation gives the time the corpse = 98.6 degF at -2.1 hours before discovery at 11am")
print("Time of death is 8:54 AM")
```

```{code-cell} ipython3

```
