# -*- coding: utf-8 -*-
import numpy as np
import numpy.matlib
import torch
from scipy.spatial.ckdtree import cKDTree
import scipy.io
import os
import itertools
import gc
from pathlib import Path
import pickle
from shutil import copyfile
import datetime
import sys

# A particle is often called a unit in this script.

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
if device == 'cuda':
    print('Using cuda')
    float_tensor = torch.cuda.FloatTensor
else:
    float_tensor = torch.FloatTensor

def dA_dt(Rac, A, E, idx, m):  # all inputs are arrays of length nUnits

    substrate = (AT - A) / (AT - A + km4)

    Aidx = A[idx]

    neighbours = torch.zeros(nUnits, device=device)
    for s in range(nUnits):
        neighbours[s] = torch.sum(k4 * Aidx[s, :m[s]] * substrate[s])       # For kun at inkludere true nearest neighbours,
                                                                            # kunne jeg også have brugt forkortet version idx og til sidst have brugt z_mask,
                                                                            # som i potential()-funktionen, istedet for at bruge hvert m i Aidx.
    if calculate_w_cargo == 1:
        result = cargo_mod * (Rac * (AT - A) / (AT - A + km1)               # Rac1 upregulates Sar2
            - A / gam1 - k1 * E * A / (A + km2)                             # basal removal and inhibition by Sec23
            # k4 * A * substrate                                            # positive feedback from own particle
            + neighbours)                                                   # positive feedback from neighbour
    else:
        result = (Rac * (AT - A) / (AT - A + km1)                           # Rac1 upregulates Sar2
                  - A / gam1 - k1 * E * A / (A + km2)                       # basal removal and inhibition by Sec23
                  + neighbours)                                             # positive feedback from neighbour
    return result


def dE_dt(A, E):
    if calculate_w_cargo == 1:
        result = 1/cargo_mod * (k2 * A * (ET - E) / (ET - E + km3) - E / gam2)
    else:
        result = k2 * A * (ET - E) / (ET - E + km3) - E / gam2
    return result

def vorticity(x, q):
    d, idx = find_potential_neighbours(x, k=30)

    # Find true neighbours:
    full_n_list = x[idx]
    dx = x[:, None, :] - full_n_list
    # Making tensors for true_neighbours to work:
    idx = torch.tensor(idx, dtype=torch.long, device=device)
    d = torch.tensor(d, dtype=torch.float, device=device)
    dx = torch.tensor(dx, dtype=torch.float, device=device)
    z_mask = find_true_neighbours(d, dx)
    # Minimize size of z_mask and reorder idx and dx
    sort_idx = torch.argsort(z_mask * 1, dim=1, descending=True)
    z_mask = torch.gather(z_mask, 1, sort_idx)
    dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
    idx_Sar = torch.gather(idx, 1, sort_idx)
    m_numOfNeigh = torch.sum(z_mask, dim=1)
    m = torch.max(m_numOfNeigh) + 1
    z_mask = z_mask[:, :m]
    dx = dx[:, :m]
    idx = idx_Sar[:, :m]
    # Normalize dx:
    d = torch.sqrt(torch.sum(dx ** 2, dim=2))
    dx = dx / d[:, :, None]

    q = torch.tensor(q, dtype=torch.float, device=device)
    qj = q[idx]

    #Removing edge units from calculation:
    no_edge_mask = m_numOfNeigh>4

    # Calculating vorticity:
    vort = torch.abs( torch.sum( (dx[:, :, 0] * qj[:, :, 1] - dx[:, :, 1] * qj[:, :, 0]) * z_mask, dim=1 ) / m_numOfNeigh * no_edge_mask)  # Do I wish absolute value?

    # finding index of units with highest vorticity:
    sort_i_vort = torch.argsort(vort, dim=0, descending=True)
    i_vortex_units = sort_i_vort[0]
    print("index of max value:", i_vortex_units)
    print("max vortex:", vort[i_vortex_units])

    return i_vortex_units


def init_gastrumech():
    global nUnits, indexes

    if startdata == 'square1384':
        P = scipy.io.loadmat('data/square1384wPCP.mat')
        P = P['p']
        leaveOutAboveUnit = 910
        x = P[:leaveOutAboveUnit, 0:3]
        p = P[:leaveOutAboveUnit, 3:6]
        q = P[:leaveOutAboveUnit, 6:9]
    if startdata == 'square910':
        filename = f'data/'+startdata
        abspath = Path(filename).absolute()
        with open(str(abspath), 'rb') as f:
            X, A_notused, E_notused, alpha_notused, beta_notused, P, Q, nSaves_notused, save_every_notused = pickle.load(f)
            x = X[-1, :, :]
            p = P[-1, :, :]
            q = Q[-1, :, :]

    nUnits = len(x[:, 0])
    indexes = torch.arange(nUnits, device=device)
    lam = np.array([l1_0, l2_0, l3_0 ])
    lam = np.matlib.repmat(lam, nUnits, 1)
    alpha = np.zeros((nUnits, 1))
    beta = np.ones((nUnits, 1))*beta0
    r = np.sqrt(np.sum(x ** 2, 1))

    # Initializing q:
    if q_start == 'curl':
        rhat = x / r[:, None]
        q = np.cross(p, rhat)
    if q_start == 'random_xyz':
        q = 2 * np.random.rand(nUnits, 3) - 1
    if q_start == 'random_xy':
        random2D = 2 * np.random.rand(nUnits, 2) - 1
        q[:, :2] = random2D
    # Normalize q:
    q /= np.sqrt(np.sum(q ** 2, 1))[:, None]

    # For Sar-feedback:
    maxTimesteps = nSaves*save_every
    X = np.zeros((nSaves+1, nUnits, len(x[0,:]) ))
    X[0, :, :] = x                                  # storing the initial positions

    # Calculate vorticity to find starting unit:
    i_vort = vorticity(x, q)

    time = np.arange(0, maxTimesteps * dt + dt, dt)
    Rac = np.zeros([maxTimesteps+1, nUnits])
    hRac = 2
    RacOneCell = time ** hRac / (time ** hRac + 2 ** hRac)
    Rac[:, i_vort] = RacOneCell # I think the central particle has index 2
    A = np.zeros(nUnits)  # Sar1
    E = np.zeros(nUnits)  # Sec
    A_storage = np.zeros((nSaves+1, nUnits))
    E_storage = np.zeros((nSaves+1, nUnits))
    alpha_storage = np.zeros((nSaves, nUnits))
    beta_storage = np.zeros((nSaves, nUnits))
    p_storage = np.zeros((nSaves, nUnits, 3))
    q_storage = np.zeros((nSaves, nUnits, 3))
    global cargo_mod
    if cargo_distri == 'gaussian':
        #cargo_mod = np.exp( - (x[:,0]**2 + x[:,1]**2) / (2*cargo_sigma**2) ) * cargo_height                            # when q curlcs around origin
        cargo_mod = np.exp( - ((x[:, 0] - x[i_vort, 0]) ** 2 + (x[:, 1] - x[i_vort, 1]) ** 2) / (2 * cargo_sigma ** 2)) * cargo_height # when q curl around i_vort
    if cargo_distri == 'uniform':
        cargo_mod = np.ones(len(x[:,0]))*cargo_height
    if cargo_distri == 'uniform_step':
        rho = np.sqrt(np.sum((x[:, 0:2] - x[i_vort, 0:2]) ** 2, axis=1))
        cargo_mod = np.ones(len(x[:,0])) * 0.6
        cargo_mod[rho < cargo_sigma] = cargo_height
    if cargo_distri == 'soft_step':
        rho = np.sqrt(np.sum((x[:, 0:2] - x[i_vort, 0:2]) ** 2, axis=1))
        hCar = 6
        cargo_mod = cargo_sigma**hCar / (rho**hCar + cargo_sigma**hCar) * cargo_height

    # The cargo concentration can not be very close to zero, because of dE_dt() where I divide with the cargo:
    #print('min(cargo_mod) before correction:', min(cargo_mod))
    small = 0.1
    if any(item < small for item in cargo_mod):
        #print(f'Some cargo_mod have very small numbers! These are set to {small}.')
        small_mask = (cargo_mod < small)
        cargo_mod[small_mask] = small
    #print('min(cargo_mod) after correction:', min(cargo_mod))

    return x, X, p, q, alpha, beta, lam, Rac, A, E, i_vort, A_storage, E_storage, alpha_storage, beta_storage, p_storage, q_storage #, nUnits


def find_potential_neighbours(x, k=100, distance_upper_bound=np.inf):
    tree = cKDTree(x)
    d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, n_jobs=-1)
    return d[:, 1:], idx[:, 1:]


def find_true_neighbours(d, dx):
    with torch.no_grad():
        z_masks = []
        i0 = 0
        batch_size = 250
        i1 = batch_size
        while True:
            if i0 >= dx.shape[0]:
                break

            n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)
            n_dis += 1000 * torch.eye(n_dis.shape[1], device=device)[None, :, :]

            z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= 0  # check summatio dimension, etc.
            z_masks.append(z_mask)

            if i1 > dx.shape[0]:
                break
            i0 = i1
            i1 += batch_size
    z_mask = torch.cat(z_masks, dim=0)
    return z_mask


def potential(x, p, q, idx, d, lam, alpha, beta, z_mask, dx, m, isotropyMask, anisotropyMask):
    # creating the arrays for later calculations:
    pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)
    pj = p[idx]
    qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)
    qj = q[idx]
    lami = lam[:, None, :].expand(p.shape[0], idx.shape[1], 3)
    lamj = lam[idx]
    # Rule for combining lambda_i and lambda_j:
    lam = (lami + lamj) * 0.5
    lam2 = torch.ceil(lami*lamj) * (lami + lamj) * 0.5
    lam[:, :, 1] = lam2[:, :, 1]
    aj = alpha[idx]
    ai = alpha[:, None, :].expand(p.shape[0], idx.shape[1], 1)

    # Introduce ptilde:
    isotropyMask = isotropyMask[:, None, None].expand(isotropyMask.shape[0], 1, 1)
    anisotropyMask = anisotropyMask[:, None, None].expand(anisotropyMask.shape[0], 1, 1)

    qmean = (qi + qj) * 0.5
    alphamean = (ai + aj) * 0.5
    qdotx = torch.sum(qmean * dx, dim=2)
    qdotx = qdotx[:, :, None].expand(p.shape[0], idx.shape[1], 1)
    alphafactor = qdotx * alphamean

    pti = pi - alphafactor * qmean * anisotropyMask - alphamean * dx * isotropyMask
    ptj = pj + alphafactor * qmean * anisotropyMask + alphamean * dx * isotropyMask

    # Normalize ptilde
    pti = pti / torch.sqrt(torch.sum(pti ** 2, dim=2))[:, :, None]
    ptj = ptj / torch.sqrt(torch.sum(ptj ** 2, dim=2))[:, :, None]

    # introducing qtilde:
    if qtilde == 1:
        bj = beta[idx]
        bi = beta[:, None, :].expand(p.shape[0], idx.shape[1], 1)
        betamean = (bi + bj) * 0.5
        #qti = qi - betamean * dx
        #qtj = qj + betamean * dx

        qti = qi - betamean * qdotx * torch.cross(pi, qi, dim=2)
        qtj = qj + betamean * qdotx * torch.cross(pj, qj, dim=2)

        qti = qti / torch.sqrt(torch.sum(qti ** 2, dim=2))[:, :, None]
        qtj = qtj / torch.sqrt(torch.sum(qtj ** 2, dim=2))[:, :, None]

        if abs_S2 != 1:
            S2 = torch.sum(torch.cross(pi, qti, dim=2) * torch.cross(ptj, qtj, dim=2), dim=2)
        if abs_S2 == 1:
            S2 = torch.abs( torch.sum(torch.cross(pi, qti, dim=2) * torch.cross(ptj, qtj, dim=2), dim=2) )

    # Calculating S:
    S1 = torch.sum(torch.cross(ptj, dx, dim=2) * torch.cross(pti, dx, dim=2), dim=2)
    if (qtilde != 1) & (abs_S2 != 1):
        S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)
    if (qtilde != 1) & (abs_S2 == 1):
        S2 = torch.abs(torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2))
    S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)

    S = lam[:, :, 0] * S1 + lam[:, :, 1] * S2 + lam[:, :, 2] * S3

    # Potential
    Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d / 5))
    V = torch.sum(Vij)

    return V, int(m)


def init_simulation(lam, p, q, x, alpha, beta, Rac, A, E):
    global gam1, gam2, AT, ET, km1, km2, km3, km4, k1, k2, k4, cargo_mod
    sqrt_dt = np.sqrt(dt)
    x = torch.tensor(x, requires_grad=True, dtype=torch.float, device=device)
    p = torch.tensor(p, requires_grad=True, dtype=torch.float, device=device)
    q = torch.tensor(q, requires_grad=True, dtype=torch.float, device=device)
    lam = torch.tensor(lam, dtype=torch.float, device=device)
    alpha = torch.tensor(alpha, dtype=torch.float, device=device)
    beta = torch.tensor(beta, dtype=torch.float, device=device)
    Rac = torch.tensor(Rac, dtype=torch.float, device=device)
    A = torch.tensor(A, dtype=torch.float, device=device)
    E = torch.tensor(E, dtype=torch.float, device=device)
    gam1 = torch.tensor(gam1, dtype=torch.float, device=device)
    gam2 = torch.tensor(gam2, dtype=torch.float, device=device)
    AT = torch.tensor(AT, dtype=torch.float, device=device)
    ET = torch.tensor(ET, dtype=torch.float, device=device)
    km1 = torch.tensor(km1, dtype=torch.float, device=device)
    km2 = torch.tensor(km2, dtype=torch.float, device=device)
    km3 = torch.tensor(km3, dtype=torch.float, device=device)
    km4 = torch.tensor(km4, dtype=torch.float, device=device)
    k1 = torch.tensor(k1, dtype=torch.float, device=device)
    k2 = torch.tensor(k2, dtype=torch.float, device=device)
    k4 = torch.tensor(k4, dtype=torch.float, device=device)
    cargo_mod = torch.tensor(cargo_mod, dtype=torch.float, device=device)
    return lam, p, q, sqrt_dt, x, alpha, beta, Rac, A, E


class TimeStepper:
    def __init__(self, init_k):
        self.k = init_k
        self.true_neighbour_max = init_k // 2
        self.d = None
        self.idx = None

    def update_k(self, true_neighbour_max, tstep):
        k = self.k
        fraction = true_neighbour_max / k
        if fraction < 0.25:
            k = int(0.75 * k)
        elif fraction > 0.75:
            k = int(1.5 * k)
        n_update = 1 if tstep < 50 else max([1, int(20 * np.tanh(tstep / 200))])
        self.k = k
        return k, n_update

    def time_step(self, lam, p, q, sqrt_dt, tstep, x, alpha, beta, Rac, A, E, i_vort):

        assert q.shape == x.shape
        assert x.shape == p.shape

        # Only update _potential_ neighbours every x steps late in simulation
        k, n_update = self.update_k(self.true_neighbour_max, tstep)
        if tstep % n_update == 0 or self.idx is None:
            d, idx = find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)
            self.idx = torch.tensor(idx, dtype=torch.long, device=device)
            self.d = torch.tensor(d, dtype=torch.float, device=device)
        idx = self.idx
        d = self.d

        # Normalise p, q
        with torch.no_grad():
            p /= torch.sqrt(torch.sum(p ** 2, dim=1))[:, None]
            q /= torch.sqrt(torch.sum(q ** 2, dim=1))[:, None]

        # Find true neighbours

        full_n_list = x[idx]
        dx = x[:, None, :] - full_n_list
        z_mask = find_true_neighbours(d, dx)
        # Minimize size of z_mask and reorder idx and dx
        sort_idx = torch.argsort(z_mask * 1, dim=1, descending=True)
        z_mask = torch.gather(z_mask, 1, sort_idx)
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))
        idx_Sar = torch.gather(idx, 1, sort_idx)
        m_Sar = torch.sum(z_mask, dim=1)
        m = torch.max(m_Sar) + 1

        z_mask = z_mask[:, :m]
        dx = dx[:, :m]
        idx = idx_Sar[:, :m]
        # Normalize dx
        d = torch.sqrt(torch.sum(dx ** 2, dim=2))
        dx = dx / d[:, :, None]

        # Updating Sar:
        q_nonzero_mask = (E < Elim)
        q_zero_Mask = (E > Elim)

        # Update the alpha and lambda independently of protein concentrations (remember to create r0 amd r1 in bottom of code):
        # rho = torch.sqrt(torch.sum(x[:, 0:2] ** 2, 1))
        #with torch.no_grad():
        #    alpha[rho > r1] = 0
        #    alpha[rho < r1] = alpha0
            # alpha[rho < r0] = 0
        #    lam[:, 0] = 1
        #    lam[:, 1] = 0
        #    lam[:, 2] = 0
        #    lam[rho < r1, 0] = l1
        #    lam[rho < r1, 1] = 1 - (l1 + l3)
        #    lam[rho < r1, 2] = l3

        if tstep % updateSar == 0 or tstep == 1:
            # Sar1/sec23 feedback circuit:
            A += dA_dt(Rac[tstep, :], A, E, idx_Sar, m_Sar) * dt  # how to apply open boundary conditions?
            E += dE_dt(A, E) * dt


            with torch.no_grad():
                for j in range(len(A)):
                    if (alpha[j, 0] < A[j]) :
                        alpha[j, 0] = A[j]  #A[j] * A[j]**20 / ( A[j]**20 + 0.6**20 )

                #beta[:, 0] = A

                lam[q_nonzero_mask, 0] = l1
                lam[q_nonzero_mask, 1] = l2
                lam[q_nonzero_mask, 2] = l3
                lam[q_zero_Mask, 0] = l1_0
                lam[q_zero_Mask, 1] = l2_0
                lam[q_zero_Mask, 2] = l3_0
                # lambda for the central input-unit is set to:
                if inputUnit_zero_q == 1:
                    for index in i_vort:
                        lam[index, 0] = l1
                        lam[index, 1] = 0
                        lam[index, 2] = 0

            alpha.requires_grad = True
            lam.requires_grad = True

        # giving the central units no q-vector by setting lam2=0 if it should be dynamic which unit is th central one:
        #i_vort = vorticity(x, q)
        #with torch.no_grad():
        #    if inputUnit_zero_q == 1:
        #        for index in i_vort:
        #            lam[index, 0] = l1
        #            lam[index, 1] = 0
        #            lam[index, 2] = 0
        #lam.requires_grad = True

        isotropyMask = (E > Elim)
        anisotropyMask = (E < Elim)

        # Calculate potential:
        V, self.true_neighbour_max = potential(x, p, q, idx, d, lam, alpha, beta, z_mask, dx, m, isotropyMask, anisotropyMask )

        # Backpropagation:
        V.backward()

        noise_mask_q = q_nonzero_mask[:, None].expand(len(q_nonzero_mask), 3)
        with torch.no_grad(): # updating x, p, and q:
            x += -x.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt
            p += -p.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt
            # q can be kept fixed by out commenting the next line
            q += -q.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt * noise_mask_q

        # Zero gradients to avoid accumulation of the gradients:
        x.grad.zero_()
        p.grad.zero_()
        q.grad.zero_()

        return x, p, q, alpha, lam


def simulation(x, p, q, alpha, beta, lam, Rac, A, E, i_vort, yield_every=50):

    lam, p, q, sqrt_dt, x, alpha, beta, Rac, A, E = init_simulation(lam, p, q, x, alpha, beta, Rac, A, E)
    time_stepper = TimeStepper(init_k=200)
    tstep = 0
    while True:
        tstep += 1
        x, p, q, alpha, lam = time_stepper.time_step(lam, p, q, sqrt_dt, tstep, x, alpha, beta, Rac, A, E, i_vort)

        if tstep % yield_every == 0:
            xx = x.detach().to("cpu").numpy()
            pp = p.detach().to("cpu").numpy()
            qq = q.detach().to("cpu").numpy()
            aa = alpha.detach().to("cpu").numpy()
            aa = aa[:, 0]
            bb = beta.detach().to("cpu").numpy()
            bb = bb[:, 0]
            AA = A.detach().to("cpu").numpy()
            EE = E.detach().to("cpu").numpy()
            yield xx, pp, qq, aa, bb, AA, EE

        gc.collect()


def main():

    x, X, p, q, alpha, beta, lam, Rac, A, E, i_vort, A_storage, E_storage, alpha_storage, beta_storage, p_storage, q_storage = init_gastrumech()

    global i
    for xx, pp, qq, aa, bb, AA, EE in itertools.islice(
            simulation(x, p, q, alpha, beta, lam,  Rac, A, E, i_vort, yield_every=save_every), nSaves):

        X[i+1, :, :] = xx
        A_storage[i+1, :] = AA
        E_storage[i+1, :] = EE
        alpha_storage[i, :] = aa
        beta_storage[i, :] = bb
        p_storage[i, :, :] = pp
        q_storage[i, :, :] = qq
        i += 1
        print(f'Running {i} of {nSaves}', end='\r')

    with open(f'myData/data_{name}', 'wb') as f:
        pickle.dump([X, A_storage, E_storage, alpha_storage, beta_storage, p_storage, q_storage, nSaves, save_every], f)

    print(f'Simulation done, saved {nSaves} datapoints')


if __name__ == '__main__':
    gam1 = 1        # half time of the basal removal for Sar1
    gam2 = 4        # half time of the basal removal for Sec
    AT = 1          # total Sar1 concentration
    ET = 1          # total Sec concentration

    # Michaelis Menten constants:
    km1 = 0.5
    km2 = 0.5
    km3 = 0.5
    km4 = 0.5

    k1 = 20  #3              # sec inhibiting sar
    k2 = 1
    k4 = 1              # in front of positive feedback loop on Sar1

    calculate_w_cargo = 1  # set to 1 to include cargo in calculations
    cargo_distri = 'soft_step' #'uniform' or 'gaussian' or 'uniform_step', or 'soft_step'
    cargo_sigma = 10         # sigma of the 2d-cargo-guassian, or width of soft step function
    cargo_height = 4         # height of 2d-cargo-gaussian or level of uniform cargo distribution or soft step function
    cargo_mod = 0            # cargo modifyer, just to define the variable before using it, can be set to anything, this is overwritten later.

    beta0 = 0.2
    # Starting condition for lambda (alpha starts wit being zero everywhere):
    l1_0 = 1
    l2_0 = 0
    l3_0 = 0  # these 3 doesn't have to sum to 1 anymore
    # new lambda setting when protein concentration increases:
    l1 = 0.5
    l2 = 0.5
    l3 = 0  # these 3 doesn't have to sum to 1 anymore
    q_start = 'curl'      #how q is initially organized. set this to 'curl', 'random_xyz', or ''random_xy' or anything else to get q parallel oriented
    np.random.seed(int(sys.argv[3]))           # int(sys.argv[2])
    torch.manual_seed(int(sys.argv[3]))

    qtilde = 1
    inputUnit_zero_q = 0    # Set this to 1 if the central input unit shouldn't be included in PCP-calculation.
    vort_limit = 0.45
    abs_S2 = 0
    eta = 0.05  # the size of this variable decides the size of the noise. Set it to 0 eliminate noise in the system.

    name = 1

    print('name:', name)

    dt = 0.025                  # choose one that gives mod(nSaves*save_every, dt) = 0
    startdata = 'square910'     # Set this variable to 'square1384' or 'square910'. 'square910' is smaller but is more a true square, it is also run to steady state with lamda1=1 everywhere.
    nSaves = 60                 # Number of saves during simulation
    save_every = 1500           # Number of time steps before saving
    print('The total amount of time steps is', nSaves * save_every)
    updateSar = 1500            # Time scale separation: how many time steps should be performed for the bending of the model between updates of protein concentration
    Elim = 0.05 #0.4 is better when k1=3. The limit when the bending becomes isotropic.

    # Saving a copy of the code:
    current_time = datetime.datetime.now()
    hour = current_time.hour + 1  # +1 for correcting for time zone differences
    date = datetime.datetime.now().date()
    copyfile(__file__, f'./script_backups/script_{name}_{date}_h{hour}_scriptname_{__file__}')

    i = 0

    main()
