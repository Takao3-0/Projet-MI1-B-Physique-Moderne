# -*- coding: utf-8 -*-
####------OndePlane1d1B-----###
#Ondes planes 1D : onde plane unique et superposition de 3 ondes planes (enveloppe) - Partie 1
#auteur : Alexandre Gourdon
#mail : contact@alexandre-gourdon.fr
#contributeur : Claude de Anthropic AI pour la relecture du code, et la conformité du sujet + creation d'un script shell et un .bat d'execution du projet. (Abonnement Claude MAX *5 Opus 4.8 Ultracode)
#date creation : 19 juin 2026
###
# Programme 1 : ondes planes 1D
#   1.1 : une onde plane Psi(x,t) = A exp(i(kx - omega t))
#   1.2 : superposition de 3 ondes planes (enveloppe)

from numpy import pi, exp, sqrt, real, imag, zeros, linspace, cos
import matplotlib.pyplot as plt


def PlaneWave(amp, k, omega, x, t):
    # onde plane A exp(i(kx - omega t)) ; x peut etre un nombre ou un tableau
    return amp * exp(1j * (k * x - omega * t))


def trace_onde_unique():
    # 1.1 : parties reelle et imaginaire d'une onde plane a t fixe
    amplitude = 1.0
    nombre_onde = 2.0
    pulsation = 3.0
    instant = 0.0

    longueur_onde = 2.0 * pi / nombre_onde
    x = linspace(0.0, 4.0 * longueur_onde, 600)

    psi = PlaneWave(amplitude, nombre_onde, pulsation, x, instant)

    fig, ax = plt.subplots()
    ax.plot(x, real(psi), label="partie reelle  Re(Psi) = A cos(kx - wt)")
    ax.plot(x, imag(psi), label="partie imaginaire  Im(Psi) = A sin(kx - wt)")
    ax.set_xlabel("position x")
    ax.set_ylabel("Psi(x, t fixe)")
    ax.set_title("Partie 1.1 : onde plane  Psi = A exp(i(kx - wt))")
    ax.legend()
    ax.grid(True)
    plt.show()


def superposition_trois_ondes(amplitude, k0, delta_k, x, t):
    # somme de 3 ondes (k0, k0-dk/2, k0+dk/2) -> enveloppe A(1 + cos(dk x/2))
    onde1 = PlaneWave(amplitude, k0, 0.0, x, t)
    onde2 = PlaneWave(amplitude / 2.0, k0 - delta_k / 2.0, 0.0, x, t)
    onde3 = PlaneWave(amplitude / 2.0, k0 + delta_k / 2.0, 0.0, x, t)
    somme = onde1 + onde2 + onde3
    enveloppe = amplitude * (1.0 + cos(delta_k * x / 2.0))
    return onde1, onde2, onde3, somme, enveloppe


def trace_superposition():
    # 1.2 : 3 ondes, leur somme et l'enveloppe sur [-pi/dk, pi/dk]
    amplitude = 1.0
    k0 = 4.0
    delta_k = 0.5

    x = linspace(-pi / delta_k, pi / delta_k, 1000)

    onde1, onde2, onde3, somme, enveloppe = superposition_trois_ondes(
        amplitude, k0, delta_k, x, 0.0)

    fig, ax = plt.subplots()
    ax.plot(x, real(onde1), linewidth=0.8, label="Re(onde 1)  k0")
    ax.plot(x, real(onde2), linewidth=0.8, label="Re(onde 2)  k0 - dk/2")
    ax.plot(x, real(onde3), linewidth=0.8, label="Re(onde 3)  k0 + dk/2")
    ax.plot(x, real(somme), color="black", linewidth=1.8, label="Re(somme)")
    ax.plot(x, enveloppe, "r--", label="enveloppe  A(1 + cos(dk x/2))")
    ax.plot(x, -enveloppe, "r--")

    ax.set_xlabel("position x")
    ax.set_ylabel("partie reelle")
    ax.set_title("Partie 1.2 : superposition de 3 ondes planes et enveloppe")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True)
    plt.show()


def main():
    choix = "tout"

    if choix == "1.1":
        trace_onde_unique()
    elif choix == "1.2":
        trace_superposition()
    elif choix == "tout":
        trace_onde_unique()
        trace_superposition()


if __name__ == "__main__":
    main()
