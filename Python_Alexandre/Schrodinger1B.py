# -*- coding: utf-8 -*-
####------Schrodinger1B-----###
#Resolution de l'equation de Schrodinger 1D (schema FTCS) et validation a V0=0 - Partie 3.2
#auteur : Alexandre Gourdon
#mail : contact@alexandre-gourdon.fr
#contributeur : Claude de Anthropic AI pour la relecture du code, et la conformité du sujet + creation d'un script shell et un .bat d'execution du projet. (Abonnement Claude MAX *5 Opus 4.8 Ultracode)
#date creation : 19 juin 2026
###
# Programme 3B : equation de Schrodinger 1D dependante du temps
#   i hbar dPsi/dt = -(hbar^2/2m) d2Psi/dx2 + V(x) Psi
# Schema explicite FTCS (Euler avant en temps, difference centree en espace).
# Le FTCS est instable a long terme : on garde r = hbar dt/(2 m dx^2) <= 0.5
# et un temps court. Validation a V0 = 0 contre la forme analytique GaussWP.
# Unites reduites : hbar = m = 1.

from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib.pyplot as plt

# Unites reduites
hbar = 1.0
m = 1.0


def GaussWP(k0, a, x, t):
    # paquet gaussien libre (forme fermee, unites reduites) : reference exacte
    n = len(x)
    psi = zeros(n, dtype=complex)
    prefacteur = (1.0 / (8.0 * pi**3)) ** 0.25
    for i in range(n):
        D = m * a**2 + 2j * hbar * t
        racine = sqrt(4.0 * pi * m * a / D)
        arg = (m / 4.0) * (a**2 * k0 + 2j * x[i])**2 / D - a**2 * k0**2 / 4.0
        psi[i] = prefacteur * racine * exp(arg)
    return psi


def norme_slice(psi_ligne, dx):
    # integrale de |psi|^2 dx sur une tranche temporelle
    somme = 0.0
    for j in range(len(psi_ligne)):
        somme = somme + abs(psi_ligne[j])**2 * dx
    return somme


def resoudre_schrodinger(x, t, V, psi0):
    # FTCS : renvoie le tableau 2D psi[j][n] (j = espace, n = temps)
    nx = len(x)
    nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    r = hbar * dt / (2.0 * m * dx**2)
    print("  pas d'espace dx =", dx, " pas de temps dt =", dt)
    print("  parametre r = hbar*dt/(2*m*dx^2) =", r, " (borne pratique : r <= 0.5)")

    psi = zeros((nx, nt), dtype=complex)
    for j in range(nx):
        psi[j][0] = psi0[j]

    for n in range(nt - 1):
        # bords fixes a zero (Dirichlet)
        psi[0][n + 1] = 0.0
        psi[nx - 1][n + 1] = 0.0
        for j in range(1, nx - 1):
            laplacien = psi[j + 1][n] - 2.0 * psi[j][n] + psi[j - 1][n]
            psi[j][n + 1] = (psi[j][n]
                             + 1j * hbar * dt / (2.0 * m * dx**2) * laplacien
                             - 1j * dt / hbar * V[j] * psi[j][n])
    return psi


def validation_particule_libre():
    # V0 = 0 : comparaison du solveur FTCS a la solution analytique GaussWP
    a = 2.0
    k0 = 2.0
    x_min, x_max = -15.0, 15.0
    nx = 301
    t_max = 1.5       # temps court (le FTCS ne reste valable que peu de temps)
    nt = 3001

    x = linspace(x_min, x_max, nx)
    t = linspace(0.0, t_max, nt)
    dx = x[1] - x[0]

    V = zeros(nx)
    for j in range(nx):
        V[j] = 0.0

    psi0 = GaussWP(k0, a, x, 0.0)

    psi = resoudre_schrodinger(x, t, V, psi0)

    print("  norme a t=0      :", norme_slice(psi[:, 0], dx))
    print("  norme a t=t_max/2:", norme_slice(psi[:, (nt - 1) // 2], dx))
    print("  norme a t=t_max  :", norme_slice(psi[:, nt - 1], dx))

    psi_ana = GaussWP(k0, a, x, t_max)

    dens_num = zeros(nx)
    dens_ana = zeros(nx)
    for j in range(nx):
        dens_num[j] = abs(psi[j][nt - 1])**2
        dens_ana[j] = abs(psi_ana[j])**2

    num = 0.0
    den = 0.0
    for j in range(nx):
        num = num + (dens_num[j] - dens_ana[j])**2
        den = den + dens_ana[j]**2
    err_rel = sqrt(num) / sqrt(den)
    print("  erreur relative L2 sur |Psi|^2 au temps final :", err_rel)

    dens_ini = zeros(nx)
    for j in range(nx):
        dens_ini[j] = abs(psi0[j])**2

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, dens_ini, "g--", label="|Psi|^2 initial (t=0)")
    ax.plot(x, dens_ana, "k-", linewidth=2, label="|Psi|^2 analytique (t_max)")
    ax.plot(x, dens_num, "r.", markersize=4, label="|Psi|^2 numerique FTCS (t_max)")
    ax.set_xlabel("position x")
    ax.set_ylabel("densite de probabilite |Psi|^2")
    ax.set_title("Validation du solveur a V0 = 0 : numerique vs analytique")
    ax.legend()
    ax.grid(True)
    plt.show()


def main():
    choix = "tout"
    if choix == "tout":
        validation_particule_libre()


if __name__ == "__main__":
    main()
