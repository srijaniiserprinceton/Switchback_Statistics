# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 19:01:59 2022
@author: naisf
"""
import sys
import numpy as np
import scipy.stats
import emcee
import matplotlib.gridspec as gridspec

from matplotlib import rc
# from chainconsumer import ChainConsumer
import corner
from matplotlib import pyplot as plt
plt.ion()

# rc('text', usetex=True)
rc('xtick', labelsize=16)
rc('ytick', labelsize=16)

#%%%
#Function definition

def two_gaussians(x, p):
    mu0  = np.array(p[0:2])
    sig0 = np.diag(np.array(p[2:4])**2)
    mu   = np.array(p[4:6])
    sig = np.diag(np.array(p[6:8])**2)
    Gauss0 = scipy.stats.multivariate_normal(mu0, sig0)
    Gauss  = scipy.stats.multivariate_normal(mu, sig)    
    if len(x.shape) >2:
        x_flat = np.array([x[0].flatten(), x[1].flatten()])
        Two_Gauss = lambda x: (1-p[8])*Gauss0.pdf(x) + p[8]*Gauss.pdf(x)
        return Two_Gauss(x_flat.T).reshape(x.shape[1:])
    else:
        return (1-p[8])*Gauss0.pdf(x) + p[8]*Gauss.pdf(x)

def log_prior(p):
    mu_phi0, mu_theta0, sig_phi0, sig_theta0, mu_phi, mu_theta, sig_phi, sig_theta, gamma =p
    if (-10 < mu_phi0 < 10. and -10.0 < mu_theta0 < 10. and .1 < sig_phi0 < 30. and .1 < sig_theta0 < 30  
        and -180.0 < mu_phi < 180. and -90.0 < mu_theta < 90. and .1 < sig_phi < 180 and .1 < sig_theta < 90
        and 0.<gamma<1.):     
        return 0.0
    return -np.inf

def log_likelihood(p, grid, data):
    Z = two_gaussians(grid, p)
    sigma_e = data.max()*0.1
    S = np.sum((data-Z)**2/sigma_e**2)
    return -S

def log_probability(p, grid, data):
    lp = log_prior(p)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(p, grid, data)    

plt.close('all')
i_fig = 1
#%%%

#generate synthetic data
P_true = np.array([1, 3, 15,9, -8, -3, 32, 25, 0.7])

x_edges = np.arange(-180,181)
y_edges = np.arange(-90,91)
x = (x_edges[1:] + x_edges[:-1])/2.
y = (y_edges[1:] + y_edges[:-1])/2.
X,Y = np.meshgrid(x,y)

data = two_gaussians(np.array([X,Y]), P_true) 
#%%%

#to set to True once run
read_only = False # True

# backend name
backend_name = 'backend.h5'

#initialize walkers
n_walkers = 32

mu_phi_0    = np.random.uniform(-10,10,n_walkers)
mu_theta_0  = np.random.uniform(-10,10,n_walkers)
sig_phi_0   = np.random.uniform(.1,30,n_walkers)
sig_theta_0 = np.random.uniform(.1,30,n_walkers)
mu_phi      = np.random.uniform(-180,180,n_walkers)
mu_theta    = np.random.uniform(-90,90,n_walkers)
sig_phi     = np.random.uniform(.1,180,n_walkers)
sig_theta   = np.random.uniform(.1,90,n_walkers)
gamma        = np.random.uniform(0,1,n_walkers)

pos = np.vstack((mu_phi_0, mu_theta_0, sig_phi_0,sig_theta_0,mu_phi,mu_theta,
                 sig_phi,sig_theta,gamma)).T
ndim = pos.shape[1]

nsteps = 2500
burnin = 1500

#run mcmc and save backend if not read_only
if not read_only:
    backend = emcee.backends.HDFBackend(backend_name)
    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_probability, backend = backend, 
        args=( np.array([X,Y]), data)
    )
    state = sampler.run_mcmc(pos, nsteps, progress=True);
        
reader = emcee.backends.HDFBackend(backend_name, read_only = read_only)

#plot walker path
fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
i_fig+=1
samples = reader.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
axes[-1].set_xlabel("step number");

#plot walker position distribution in 9D space
samples = reader.get_chain(discard = burnin, flat = True)

params=[r"$\mu_{0\phi}$", r"$\mu_{0\theta}$",
     r"$\sigma_{0\phi}$", r"$\sigma_{0\theta}$",
     r"$\mu_{\phi}$", r"$\mu_{\theta}$",
     r"$\sigma_{\phi}$", r"$\sigma_{\theta}$",
     r"$\gamma$"]


'''
c = ChainConsumer()
c.add_chain(samples, parameters=params)
c.configure(summary=True, sigmas=np.linspace(0, 2, 5), shade_gradient=1.5, 
            shade_alpha=.5, cloud = True, num_cloud = 5000, kde = 1.)

fig = c.plotter.plot(truth = P_true, legend=True)
'''

# using corner plot
# tau = sampler.get_autocorr_time()
# burnin = int(2 * np.max(tau))
# thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True)#, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True)#, thin=thin)
# log_prior_samples = sampler.get_blobs(discard=burnin, flat=True)#, thin=thin)

print("burn-in: {0}".format(burnin))
# print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
# print("flat log prior shape: {0}".format(log_prior_samples.shape))

labels = list(map(r"$\theta_{{{0}}}$".format, range(1, ndim + 1)))
labels += ["log prob"]

corner.corner(samples)#, labels=labels)


sys.exit()

i_fig+=1

#print results
results = np.zeros((ndim,3))
for i in range(ndim):
    results[i] = np.array(c.analysis.get_summary()[params[i]])
results= results[:,1]

table = c.analysis.get_latex_table()
print(table)

#%%%
#plot the synthetic data

nb_levels = 1000
levels = np.linspace(data.max()/nb_levels, data.max(), nb_levels)
police = 14

fig = plt.figure(i_fig, figsize = (15,9))
i_fig +=1
gs = gridspec.GridSpec(3,6)

ax1 = fig.add_subplot(gs[1:3,0:5])
im = ax1.contourf(x,y,data, levels = levels, cmap = 'gnuplot')
ax1.set_xlabel(r'$\phi$', fontsize=police)
ax1.set_ylabel(r'$\theta$', fontsize=police)
ax1.tick_params(labelsize=police)
ax1.axhline(0,c='grey')
ax1.axvline(0,c='grey')
 
    
ax11  = fig.add_subplot(gs[0,0:5], sharex = ax1)
ax11.fill_between( x, np.sum(data, axis=0), color = 'grey', alpha=.2)
ax11.tick_params('x',labelcolor='white')
ax11.set_ylabel(r'f($\phi$)', fontsize=police)   
ax11.axvline(0,c='grey', linewidth = .5)

ax12  = fig.add_subplot(gs[1:3,5], sharey = ax1)
ax12.fill_between(np.sum(data, axis=1), y, color = 'grey', alpha=.2)
ax12.tick_params('y',labelcolor='white')    
ax12.axhline(0,c='grey', linewidth = .5)
ax12.set_xlabel(r'f($\theta$)', fontsize=police)

ax1.set_xlim(-180,180)
ax1.set_ylim(-90,90)

#%%
#visualize the fit

lw = .3
alpha = .08

#draw 100 curves randomly from the parameter distribution
draw = samples.copy()
np.random.shuffle(draw)
draw = draw[:100,:]   

for i in (range(len(draw[:,0]))):
    Fit = two_gaussians(np.array([X,Y]), draw[i])
    G0 = two_gaussians(np.array([X,Y]), np.hstack((draw[i][:-1],np.array([0]))))
    G1 = two_gaussians(np.array([X,Y]), np.hstack((draw[i][:-1],np.array([1]))))
    
    ax11.plot(x, np.sum(Fit, axis = 0),color = 'silver', alpha=alpha, linewidth = 1)
    ax12.plot(np.sum(Fit, axis = 1),y,color = 'silver', alpha = alpha, linewidth = 1)
    
    ax11.plot(x, np.sum((1-draw[i][-1])*G0, axis = 0),color = 'lightsalmon', alpha=alpha, linewidth = 1)
    ax11.plot(x, np.sum(draw[i][-1]*G1, axis = 0),color = 'cornflowerblue', alpha=alpha, linewidth = 1)
    
    ax12.plot(np.sum((1-P_true[-1])*G0, axis = 1), y, color = 'lightsalmon', alpha = alpha, linewidth = 1)
    ax12.plot(np.sum(P_true[-1]*G1, axis = 1), y, color = 'cornflowerblue', alpha = alpha, linewidth = 1)
      

Fit = two_gaussians(np.array([X,Y]), results)
G0 = two_gaussians(np.array([X,Y]), np.hstack((results[:-1],np.array([0]))))
G1 = two_gaussians(np.array([X,Y]), np.hstack((results[:-1],np.array([1]))))

im = ax1.contour(x,y,Fit,colors = 'white')
ax11.plot(x, np.sum((1-results[-1])*G0, axis = 0),color = 'r', label = 'fit core ', linewidth = .5)
ax11.plot(x, np.sum(results[-1]*G1, axis = 0),color = 'b', label = 'fit SB',linewidth = .5)
ax11.plot(x, np.sum(Fit, axis = 0),color = 'k',  label = 'Fit ', linewidth = .5)
ax11.legend()

ax12.plot(np.sum((1-results[-1])*G0, axis = 1), y, color = 'r',  label = 'fit core ', linewidth = .5)
ax12.plot(np.sum(results[-1]*G1, axis = 1), y, color = 'b',  label = 'fit SB', linewidth = .5)
ax12.plot(np.sum(Fit, axis = 1),y,color = 'k',  label = 'fit ', linewidth = .5)

ax11.axvline(results[0], c='r', alpha = .5)
ax12.axhline(results[1], c='r',alpha = .5)
ax11.axvline(results[4], c='b', alpha = .5)
ax12.axhline(results[5], c='b', alpha = .5)

ax11.text(-170,np.sum(Fit, axis = 0).max()*0.2,
         "MCMC estimates: \n" +
         r'$\mu_{0\phi}$      = ' + str(np.round(results[0],1)) + '\n' +
          r'$\mu_{0\theta}$    = ' + str(np.round(results[1],1)) + '\n' +
         r'$\sigma_{0\phi}$   = ' + str(np.round(results[2],1)) + '\n' +
         r'$\sigma_{0\theta}$ = ' + str(np.round(results[3],1)) + '\n' +
          r'$\mu_{\phi}$      = ' + str(np.round(results[4],1)) + '\n' +
          r'$\mu_{\theta}$    = ' + str(np.round(results[5],1)) + '\n' +
          r'$\sigma_{\phi}$   = ' + str(np.round(results[6],1)) + '\n' +
          r'$\sigma_{\theta}$ = ' + str(np.round(results[7],1)) + '\n' +
          # r'$\rho$            = ' + str(np.round(results[8],2)) + '\n' 
          r'$\gamma$               = ' + str(np.round(results[8],2)) + '\n'
         )
         
print("MCMC estimates:")
print(r'$\mu_{0\phi}$'          + ' = {0:.3f}'.format(results[0]))
print(r'$\mu_{0\theta}$'        + ' = {0:.3f}'.format(results[1]))
print(r'$\sigma_{0\phi}$'       + ' = {0:.3f}'.format(results[2]))
print(r'$\sigma_{0\theta}$'     + ' = {0:.3f}'.format(results[3]))
print(r'$\mu_{\phi}$'           + ' = {0:.3f}'.format(results[4]))
print(r'$\mu_{\theta}$'         + ' = {0:.3f}'.format(results[5]))
print(r'$\sigma_{\phi}$'        + ' = {0:.3f}'.format(results[6]))
print(r'$\sigma_{\theta}$'      + ' = {0:.3f}'.format(results[7]))
# print(r'$correlation$       = {0:.3f}'.format(results[8]))
print(r'$\gamma$        = {0:.3f}'.format(results[8]))