# -*- coding: utf-8 -*-
####------ProjetEffetTunnel1B-----###
#Effet tunnel d'un paquet d'ondes sur une barriere (schema de Visscher) + temps de traversee - Partie 4
#auteur : Alexandre Gourdon
#mail : contact@alexandre-gourdon.fr
#contributeur : Claude de Anthropic AI pour la relecture du code, et la conformité du sujet + creation d'un script shell et un .bat d'execution du projet. (Abonnement Claude MAX *5 Opus 4.8 Ultracode)
#date creation : 19 juin 2026
###
# Sources (consultees le 16 juin 2026) :
#   - P. B. Visscher, "A fast explicit algorithm for the time-dependent
#     Schrodinger equation", Computers in Physics 5, 596 (1991).
#     https://pubs.aip.org/aip/cip/article/5/6/596/279764/A-fast-explicit-algorithm-for-the-time-dependent

from numpy import pi, exp, sqrt, real, imag, zeros, linspace, sin, sinh
import matplotlib.pyplot as plt

# Unites reduites
hbar = 1.0
m = 1.0


def construire_paquet(x, x0, a, k0):
    # paquet gaussien centre en x0, allant vers la droite (k0 > 0), normalise
    nx = len(x)
    dx = x[1] - x[0]
    psi = zeros(nx, dtype=complex)
    for j in range(nx):
        enveloppe = exp(-((x[j] - x0)**2) / a**2)
        porteuse = exp(1j * k0 * x[j])
        psi[j] = enveloppe * porteuse
    somme = 0.0
    for j in range(nx):
        somme = somme + abs(psi[j])**2 * dx
    facteur = 1.0 / sqrt(somme)
    for j in range(nx):
        psi[j] = psi[j] * facteur
    return psi


def construire_barriere(x, debut, largeur, V0):
    # V = V0 entre debut et debut+largeur, 0 ailleurs
    nx = len(x)
    V = zeros(nx)
    fin = debut + largeur
    for j in range(nx):
        if x[j] < debut:
            V[j] = 0.0
        elif x[j] <= fin:
            V[j] = V0
        else:
            V[j] = 0.0
    return V


def applique_H(f, V, dx):
    # H[f]_j = -(1/2)(f[j+1] - 2 f[j] + f[j-1])/dx^2 + V[j] f[j] ; bords a zero
    nx = len(f)
    Hf = zeros(nx)
    for j in range(1, nx - 1):
        cinetique = -0.5 * (f[j + 1] - 2.0 * f[j] + f[j - 1]) / dx**2
        Hf[j] = cinetique + V[j] * f[j]
    return Hf


def simuler(x, V, psi0, dt, nt, x_sortie, nb_images=6):
    # schema de Visscher : R aux pas entiers, I aux pas demi-entiers.
    # renvoie temps, norme, <x>, x_max, P_transmise, <x>_transmis, images, t_images
    nx = len(x)
    dx = x[1] - x[0]

    R = zeros(nx)
    I = zeros(nx)
    for j in range(nx):
        R[j] = real(psi0[j])
        I[j] = imag(psi0[j])

    temps = zeros(nt)
    norme_serie = zeros(nt)
    xmoy_serie = zeros(nt)
    xmax_serie = zeros(nt)     # position du maximum de |Psi|^2
    ptrans_serie = zeros(nt)
    xtrans_serie = zeros(nt)   # position moyenne de la partie transmise

    images = []
    temps_images = []
    if nb_images > 1:
        pas_image = (nt - 1) // (nb_images - 1)
    else:
        pas_image = nt

    for n in range(nt):
        # avance I d'un demi-pas : I^{n+1/2} = I^{n-1/2} - dt H[R^n]
        HR = applique_H(R, V, dx)
        I_suivant = zeros(nx)
        for j in range(1, nx - 1):
            I_suivant[j] = I[j] - dt * HR[j]

        # densite symetrique de Visscher : rho = R^2 + I^{n-1/2} I^{n+1/2}
        nrm = 0.0
        somme_x = 0.0
        proba_droite = 0.0
        somme_x_droite = 0.0
        densite_max = -1.0
        j_max = 0
        for j in range(nx):
            densite = R[j] * R[j] + I[j] * I_suivant[j]
            nrm = nrm + densite * dx
            somme_x = somme_x + x[j] * densite * dx
            if densite > densite_max:
                densite_max = densite
                j_max = j
            if x[j] >= x_sortie:
                proba_droite = proba_droite + densite * dx
                somme_x_droite = somme_x_droite + x[j] * densite * dx
        temps[n] = n * dt
        norme_serie[n] = nrm
        xmoy_serie[n] = somme_x / nrm
        xmax_serie[n] = x[j_max]
        ptrans_serie[n] = proba_droite / nrm
        if proba_droite > 1.0e-6:
            xtrans_serie[n] = somme_x_droite / proba_droite
        else:
            xtrans_serie[n] = x_sortie

        # enregistrement d'une image de la densite de temps en temps
        if pas_image > 0 and n % pas_image == 0:
            image = zeros(nx)
            for j in range(nx):
                image[j] = R[j] * R[j] + I[j] * I_suivant[j]
            images.append(image)
            temps_images.append(n * dt)

        # avance R d'un pas : R^{n+1} = R^n + dt H[I^{n+1/2}]
        if n < nt - 1:
            HI = applique_H(I_suivant, V, dx)
            R_suivant = zeros(nx)
            for j in range(1, nx - 1):
                R_suivant[j] = R[j] + dt * HI[j]
            for j in range(nx):
                R[j] = R_suivant[j]
                I[j] = I_suivant[j]

    return (temps, norme_serie, xmoy_serie, xmax_serie, ptrans_serie,
            xtrans_serie, images, temps_images)


def temps_de_passage(temps, signal, seuil):
    # premier instant ou signal atteint seuil (interpolation lineaire), sinon None
    for i in range(1, len(signal)):
        if (signal[i - 1] < seuil) and (signal[i] >= seuil):
            t1, t2 = temps[i - 1], temps[i]
            s1, s2 = signal[i - 1], signal[i]
            return t1 + (seuil - s1) * (t2 - t1) / (s2 - s1)
    return None


def transmission_analytique(E, V0, a):
    # coefficient de transmission (onde plane d'energie E) : 3 regimes
    if E < V0:
        kappa = sqrt(2.0 * m * (V0 - E)) / hbar
        sh = sinh(kappa * a)
        return 1.0 / (1.0 + (V0**2 * sh * sh) / (4.0 * E * (V0 - E)))
    elif E > V0:
        kp = sqrt(2.0 * m * (E - V0)) / hbar
        s = sin(kp * a)
        return 1.0 / (1.0 + (V0**2 * s * s) / (4.0 * E * (E - V0)))
    else:
        return 1.0 / (1.0 + m * V0 * a * a / (2.0 * hbar**2))


# Parametres communs (unites reduites)
A_PAQUET = 4.0         # largeur du paquet
K0 = 2.0               # nombre d'onde central
X0 = -13.0             # position initiale du centre (a gauche)
X_MIN, X_MAX = -30.0, 30.0
NX = 301               # -> dx = 0.2
T_MAX = 12.0
NT = 3001              # -> dt = 4e-3
DEBUT_BARRIERE = 0.0

E0 = hbar**2 * K0**2 / (2.0 * m)   # energie centrale du paquet


def grille():
    x = linspace(X_MIN, X_MAX, NX)
    dt = T_MAX / (NT - 1)
    return x, dt


def partie_A_visualisation():
    # visualisation de l'effet tunnel (une barriere, E < V0)
    x, dt = grille()
    V0 = 3.0
    largeur = 1.0
    fin = DEBUT_BARRIERE + largeur

    print("  energie du paquet E0 =", E0, " hauteur V0 =", V0,
          " (E0 < V0 : classiquement REFLECHI)")

    V = construire_barriere(x, DEBUT_BARRIERE, largeur, V0)
    psi0 = construire_paquet(x, X0, A_PAQUET, K0)
    temps, norme_s, xmoy_s, xmax_s, ptrans_s, xtrans_s, images, t_images = simuler(
        x, V, psi0, dt, NT, fin, nb_images=6)

    T_num = ptrans_s[NT - 1]
    T_th = transmission_analytique(E0, V0, largeur)
    print("  transmission numerique T_num (du paquet) =", T_num)
    print("  transmission analytique (onde plane a k0) T_th =", T_th)
    print("  norme finale (controle stabilite Visscher) :", norme_s[NT - 1])

    fig, ax = plt.subplots(figsize=(9, 5))
    hauteur_max = 0.0
    for image in images:
        for j in range(len(x)):
            if image[j] > hauteur_max:
                hauteur_max = image[j]
    barriere_dessin = zeros(len(x))
    for j in range(len(x)):
        if V[j] > 0:
            barriere_dessin[j] = hauteur_max
    ax.fill_between(x, 0, barriere_dessin, color="lightgray",
                    label="barriere (V0=%.1f, a=%.1f)" % (V0, largeur))
    for k in range(len(images)):
        ax.plot(x, images[k], label="t = %.2f" % t_images[k])
    ax.set_xlabel("position x")
    ax.set_ylabel("|Psi|^2")
    ax.set_title("Effet tunnel : le paquet frappe la barriere et se scinde "
                 "(reflechi + transmis)")
    ax.legend(fontsize=8)
    ax.grid(True)
    fig.savefig("fig18_interaction.png", dpi=200, bbox_inches="tight")
    plt.show()


def partie_B_temps():
    # mesure des temps tau0 (libre) et du retard du paquet transmis
    x, dt = grille()
    V0 = 4.0
    largeur = 1.0
    fin = DEBUT_BARRIERE + largeur
    x_det = 7.0                # detecteur fixe, a droite de la barriere

    V_libre = construire_barriere(x, DEBUT_BARRIERE, largeur, 0.0)
    psi0 = construire_paquet(x, X0, A_PAQUET, K0)
    t_l, norme_l, xmoy_l, xmax_l, ptrans_l, xtrans_l, img_l, ti_l = simuler(
        x, V_libre, psi0, dt, NT, fin, nb_images=2)

    tau0_th = largeur * m / (hbar * K0)        # = a / v_g

    # (1) mesure via <x>
    t_entree_libre = temps_de_passage(t_l, xmoy_l, DEBUT_BARRIERE)
    t_sortie_libre = temps_de_passage(t_l, xmoy_l, fin)
    if (t_entree_libre is None) or (t_sortie_libre is None):
        tau0_num = float("nan")
    else:
        tau0_num = t_sortie_libre - t_entree_libre
        print("  tau0_num (paquet libre, <x>, parcours de a)        =", tau0_num)

    # (2) mesure via le maximum de |Psi|^2
    t_entree_max = temps_de_passage(t_l, xmax_l, DEBUT_BARRIERE)
    t_sortie_max = temps_de_passage(t_l, xmax_l, fin)
    if (t_entree_max is None) or (t_sortie_max is None):
        tau0_num_max = float("nan")
    else:
        tau0_num_max = t_sortie_max - t_entree_max
    print("  tau0_num (paquet libre, max de |Psi|^2, parcours de a) =", tau0_num_max)
    print("  tau0_th  = a m /(hbar k0) = a / v_g                    =", tau0_th)

    t_arr_libre = temps_de_passage(t_l, xmoy_l, x_det)

    V = construire_barriere(x, DEBUT_BARRIERE, largeur, V0)
    psi0b = construire_paquet(x, X0, A_PAQUET, K0)
    t_b, norme_b, xmoy_b, xmax_b, ptrans_b, xtrans_b, img_b, ti_b = simuler(
        x, V, psi0b, dt, NT, fin, nb_images=2)

    T_num = ptrans_b[NT - 1]
    t_arr_trans = temps_de_passage(t_b, xtrans_b, x_det)
    print("  fraction transmise finale T_num =", T_num)
    print("  arrivee au detecteur x_det =", x_det)
    print("    paquet libre    : t =", t_arr_libre)
    print("    paquet transmis : t =", t_arr_trans)
    if (t_arr_trans is not None) and (t_arr_libre is not None):
        retard = t_arr_trans - t_arr_libre
        print("  retard du paquet transmis (t_trans - t_libre) =", retard)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    ax1.plot(t_l, xmoy_l, label="<x>(t) paquet libre")
    ax1.plot(t_b, xmoy_b, label="<x>(t) avec barriere (tout le paquet)")
    ax1.plot(t_b, xtrans_b, "g--", label="<x>(t) de la partie transmise")
    ax1.axhline(DEBUT_BARRIERE, color="gray", linestyle=":")
    ax1.axhline(fin, color="gray", linestyle=":")
    ax1.axhline(x_det, color="red", linestyle=":", label="detecteur x_det")
    ax1.set_xlabel("temps t"); ax1.set_ylabel("position moyenne <x>")
    ax1.set_title("Suivi du centre du paquet")
    ax1.legend(fontsize=8); ax1.grid(True)

    ax2.plot(t_b, ptrans_b, color="green",
             label="P transmise(t) (a droite de la barriere)")
    ax2.axhline(T_num, color="green", linestyle=":")
    ax2.set_xlabel("temps t"); ax2.set_ylabel("probabilite transmise")
    ax2.set_title("Montee de la probabilite transmise vers son plateau T")
    ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    fig.savefig("fig19_temps_tunnel.png", dpi=200, bbox_inches="tight")
    plt.show()


def partie_C_influence_largeur():
    # influence de la largeur a (V0 fixe)
    x, dt = grille()
    V0 = 4.0
    print("=== Partie C : influence de la largeur a (V0 =", V0, ") ===")

    x_det = 7.0   # detecteur fixe
    largeurs = [0.5, 1.0, 1.5, 2.0, 2.5]
    T_num_liste = []
    T_th_liste = []
    retard_liste = []

    V_libre = construire_barriere(x, DEBUT_BARRIERE, 1.0, 0.0)
    psi0 = construire_paquet(x, X0, A_PAQUET, K0)
    t_l, norme_l, xmoy_l, xmax_l, ptrans_l, xtrans_l, img_l, ti_l = simuler(
        x, V_libre, psi0, dt, NT, DEBUT_BARRIERE, nb_images=2)
    t_arr_libre = temps_de_passage(t_l, xmoy_l, x_det)

    for largeur in largeurs:
        fin = DEBUT_BARRIERE + largeur
        V = construire_barriere(x, DEBUT_BARRIERE, largeur, V0)
        psi0b = construire_paquet(x, X0, A_PAQUET, K0)
        t_b, norme_b, xmoy_b, xmax_b, ptrans_b, xtrans_b, img_b, ti_b = simuler(
            x, V, psi0b, dt, NT, fin, nb_images=2)
        T_num = ptrans_b[NT - 1]
        T_th = transmission_analytique(E0, V0, largeur)
        t_arr_trans = temps_de_passage(t_b, xtrans_b, x_det)
        if (t_arr_trans is not None) and (t_arr_libre is not None):
            retard = t_arr_trans - t_arr_libre
        else:
            retard = float("nan")
        T_num_liste.append(T_num)
        T_th_liste.append(T_th)
        retard_liste.append(retard)
        print("  a =", largeur, " T_num =", round(T_num, 4),
              " T_th =", round(T_th, 4), " retard =", round(retard, 3))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogy(largeurs, T_num_liste, "ro-", label="T numerique (paquet)")
    ax1.semilogy(largeurs, T_th_liste, "k.--", label="T analytique (onde plane k0)")
    ax1.set_xlabel("largeur a de la barriere"); ax1.set_ylabel("transmission T")
    ax1.set_title("La transmission chute (~ exponentiellement) avec a")
    ax1.legend(); ax1.grid(True, which="both")

    ax2.plot(largeurs, retard_liste, "bs-")
    ax2.set_xlabel("largeur a de la barriere")
    ax2.set_ylabel("retard du paquet transmis (t_trans - t_libre)")
    ax2.set_title("Le temps tunnel ne croit pas comme a/v (filtrage + Hartman)")
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig("fig20_hartman.png", dpi=200, bbox_inches="tight")
    plt.show()


def partie_D_influence_hauteur():
    # influence de la hauteur V0 (largeur fixe), inclut E > V0
    x, dt = grille()
    largeur = 1.0
    fin = DEBUT_BARRIERE + largeur
    x_det = 7.0
    print("=== Partie D : influence de la hauteur V0 (a =", largeur,
          ", E0 =", E0, ") ===")

    hauteurs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    T_num_liste = []
    T_th_liste = []
    retard_liste = []

    V_libre = construire_barriere(x, DEBUT_BARRIERE, largeur, 0.0)
    psi0 = construire_paquet(x, X0, A_PAQUET, K0)
    t_l, norme_l, xmoy_l, xmax_l, ptrans_l, xtrans_l, img_l, ti_l = simuler(
        x, V_libre, psi0, dt, NT, fin, nb_images=2)
    t_arr_libre = temps_de_passage(t_l, xmoy_l, x_det)

    for V0 in hauteurs:
        V = construire_barriere(x, DEBUT_BARRIERE, largeur, V0)
        psi0b = construire_paquet(x, X0, A_PAQUET, K0)
        t_b, norme_b, xmoy_b, xmax_b, ptrans_b, xtrans_b, img_b, ti_b = simuler(
            x, V, psi0b, dt, NT, fin, nb_images=2)
        T_num = ptrans_b[NT - 1]
        T_th = transmission_analytique(E0, V0, largeur)
        t_arr_trans = temps_de_passage(t_b, xtrans_b, x_det)
        if (t_arr_trans is not None) and (t_arr_libre is not None):
            retard = t_arr_trans - t_arr_libre
        else:
            retard = float("nan")
        T_num_liste.append(T_num)
        T_th_liste.append(T_th)
        retard_liste.append(retard)
        if E0 > V0:
            regime = "E>V0 (classiquement passe)"
        elif E0 < V0:
            regime = "E<V0 (effet tunnel)"
        else:
            regime = "E=V0 (cas limite, T=1/2)"
        print("  V0 =", V0, " T_num =", round(T_num, 4),
              " T_th =", round(T_th, 4), " retard =", round(retard, 3),
              " ->", regime)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(hauteurs, T_num_liste, "ro-", label="T numerique")
    ax1.plot(hauteurs, T_th_liste, "k.--", label="T analytique (k0)")
    ax1.axvline(E0, color="blue", linestyle=":", label="V0 = E0 (frontiere)")
    ax1.set_xlabel("hauteur V0 de la barriere"); ax1.set_ylabel("transmission T")
    ax1.set_title("Transmission vs hauteur (a gauche E>V0, a droite tunnel)")
    ax1.legend(); ax1.grid(True)

    ax2.plot(hauteurs, retard_liste, "bs-")
    ax2.axvline(E0, color="blue", linestyle=":", label="V0 = E0 (frontiere)")
    ax2.set_xlabel("hauteur V0 de la barriere")
    ax2.set_ylabel("retard du paquet transmis (t_trans - t_libre)")
    ax2.set_title("Temps de traversee vs hauteur V0")
    ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    plt.show()


def partie_E_comparaison_classique():
    # comparaison quantique / classique
    largeur = 1.0
    print("  Energie du paquet : E0 =", E0)
    for V0 in [1.0, 4.0]:
        T_th = transmission_analytique(E0, V0, largeur)
        print("  --- V0 =", V0, "---")
        if E0 > V0:
            print("    Quantiquement : T =", round(T_th, 4),
                  "(< 1 : il existe une probabilite de REFLEXION).")
        else:
            print("    Quantiquement : T =", round(T_th, 4),
                  "(> 0 : EFFET TUNNEL, la particule peut traverser).")


def main():
    choix = "tout"

    if choix == "A":
        partie_A_visualisation()
    elif choix == "B":
        partie_B_temps()
    elif choix == "C":
        partie_C_influence_largeur()
    elif choix == "D":
        partie_D_influence_hauteur()
    elif choix == "E":
        partie_E_comparaison_classique()
    elif choix == "tout":
        partie_A_visualisation()
        partie_B_temps()
        partie_C_influence_largeur()
        partie_D_influence_hauteur()
        partie_E_comparaison_classique()


if __name__ == "__main__":
    main()
