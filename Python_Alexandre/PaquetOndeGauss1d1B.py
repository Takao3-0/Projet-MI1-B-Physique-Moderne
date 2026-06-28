# -*- coding: utf-8 -*-
####------PaquetOndeGauss1d1B-----###
#Paquet d'ondes gaussien 1D (forme fermee Psi(x,t)) : profil a t=0 et etalement - Partie 2
#auteur : Alexandre Gourdon
#mail : contact@alexandre-gourdon.fr
#contributeur : Claude de Anthropic AI pour la relecture du code, et la conformité du sujet + creation d'un script shell et un .bat d'execution du projet. (Abonnement Claude MAX *5 Opus 4.8 Ultracode)
#date creation : 19 juin 2026
###

from numpy import pi, exp, sqrt, real, imag, zeros, linspace
import matplotlib.pyplot as plt

# Constantes physiques (SI) - electron
hbar = 1.054571817e-34   # Planck reduite (J.s)
m = 9.1093837015e-31     # masse de l'electron (kg)


def GaussWP(k0, a, x, t):
    # paquet gaussien libre, forme fermee Psi(x,t)
    # k0 : nombre d'onde central, a : largeur, x : tableau, t : instant
    n = len(x)
    psi = zeros(n, dtype=complex)

    prefacteur = (1.0 / (8.0 * pi**3)) ** 0.25

    for i in range(n):
        D = m * a**2 + 2j * hbar * t
        racine = sqrt(4.0 * pi * m * a / D)
        arg = (m / 4.0) * (a**2 * k0 + 2j * x[i])**2 / D - a**2 * k0**2 / 4.0
        psi[i] = prefacteur * racine * exp(arg)

    return psi


def norme(psi, dx):
    # integrale de |psi|^2 dx (somme de Riemann)
    somme = 0.0
    for i in range(len(psi)):
        somme = somme + abs(psi[i])**2 * dx
    return somme


def trace_paquet_t0():
    # profil du paquet a t = 0 + verification de la normalisation
    a = 1.0e-9          # largeur (1 nm)
    k0 = 1.0e10         # nombre d'onde central (m^-1)
    t = 0.0

    x = linspace(-5.0 * a, 5.0 * a, 1000)
    dx = x[1] - x[0]

    # controle anti-repliement : assez de points par longueur d'onde porteuse
    longueur_onde_porteuse = 2.0 * pi / k0
    pts_par_oscillation = longueur_onde_porteuse / dx
    print("Points par longueur d'onde de la porteuse =", pts_par_oscillation,
          "(garder >> 10 pour eviter le repliement)")

    psi = GaussWP(k0, a, x, t)

    n2 = norme(psi, dx)
    print("Verification de la norme a t=0 : integrale |Psi|^2 dx =", n2)

    densite = zeros(len(x))
    for i in range(len(x)):
        densite[i] = abs(psi[i])**2

    x_nm = x * 1.0e9

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))

    ax1.plot(x_nm, real(psi), label="Re(Psi)")
    ax1.plot(x_nm, imag(psi), label="Im(Psi)")
    ax1.set_xlabel("position x (nm)")
    ax1.set_ylabel("Psi  (m^-1/2)")
    ax1.set_title("Paquet d'ondes gaussien a t = 0 : parties reelle et imaginaire")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(x_nm, densite, color="black")
    ax2.set_xlabel("position x (nm)")
    ax2.set_ylabel("|Psi|^2  (m^-1)")
    ax2.set_title("Densite de probabilite |Psi|^2 (aire = 1)")
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig("fig12_paquet_t0.png", dpi=200, bbox_inches="tight")
    plt.show()


def trace_etalement():
    # le paquet libre se deplace (v_g = hbar k0/m) et s'etale au cours du temps
    a = 1.0e-9
    k0 = 1.0e10
    x = linspace(-20.0e-9, 20.0e-9, 1500)
    dx = x[1] - x[0]

    instants = [0.0, 0.5e-14, 1.0e-14]

    fig, ax = plt.subplots(figsize=(9, 5))
    for t in instants:
        psi = GaussWP(k0, a, x, t)
        densite = zeros(len(x))
        for i in range(len(x)):
            densite[i] = abs(psi[i])**2
        ax.plot(x * 1.0e9, densite, label="t = %.1e s" % t)
        print("  t = %.2e s : norme =" % t, norme(psi, dx),
              " (doit rester ~ 1 : la norme se conserve dans le temps)")
    ax.set_xlabel("position x (nm)")
    ax.set_ylabel("|Psi|^2  (m^-1)")
    ax.set_title("Le paquet gaussien libre se deplace (v_g) et s'etale")
    ax.legend()
    ax.grid(True)
    plt.show()


def main():
    choix = "tout"

    if choix == "t0":
        trace_paquet_t0()
    elif choix == "etalement":
        trace_etalement()
    elif choix == "tout":
        trace_paquet_t0()
        trace_etalement()


if __name__ == "__main__":
    main()
