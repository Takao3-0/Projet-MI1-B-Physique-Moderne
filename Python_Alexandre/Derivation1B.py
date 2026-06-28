# -*- coding: utf-8 -*-
####------Derivation1B-----###
#Derivation numerique par differences finies (derivees 1ere et 2nde) - Partie 3.1
#auteur : Alexandre Gourdon
#mail : contact@alexandre-gourdon.fr
#contributeur : Claude de Anthropic AI pour la relecture du code, et la conformité du sujet + creation d'un script shell et un .bat d'execution du projet. (Abonnement Claude MAX *5 Opus 4.8 Ultracode)
#date creation : 19 juin 2026
###
# Programme 3A : derivation numerique par differences finies (1D)
#   f'(x)  ~ (f[j+1] - f[j-1]) / (2 dx)         (centre, ordre 2)
#   f''(x) ~ (f[j+1] - 2 f[j] + f[j-1]) / dx^2  (centre, ordre 2)
#   bords : formules decentrees

from numpy import pi, exp, sqrt, real, imag, zeros, linspace, sin, cos
import matplotlib.pyplot as plt


def carre(x):
    # f(x) = x^2
    return x * x


def deux_x(x):
    # f'(x) = 2 x
    return 2.0 * x


def deux_constant(x):
    # f''(x) = 2 (constante ; 0*x pour garder la taille du tableau)
    return 2.0 + 0.0 * x


def derivee_premiere(f, dx):
    # interieur : centre (ordre 2) ; bords : avant/arriere (ordre 1)
    n = len(f)
    df = zeros(n)
    for i in range(n):
        if i == 0:
            df[i] = (f[i + 1] - f[i]) / dx
        elif i == n - 1:
            df[i] = (f[i] - f[i - 1]) / dx
        else:
            df[i] = (f[i + 1] - f[i - 1]) / (2.0 * dx)
    return df


def derivee_seconde(f, dx):
    # interieur : centre 3 points (ordre 2) ; bords : decentre 3 points
    n = len(f)
    d2f = zeros(n)
    for i in range(n):
        if i == 0:
            d2f[i] = (f[i] - 2.0 * f[i + 1] + f[i + 2]) / dx**2
        elif i == n - 1:
            d2f[i] = (f[i] - 2.0 * f[i - 1] + f[i - 2]) / dx**2
        else:
            d2f[i] = (f[i + 1] - 2.0 * f[i] + f[i - 1]) / dx**2
    return d2f


def erreur_relative_max(approx, exact):
    # plus grande erreur relative (on ignore les points ou exact ~ 0)
    err_max = 0.0
    for i in range(len(approx)):
        if abs(exact[i]) > 1.0e-12:
            err = abs(approx[i] - exact[i]) / abs(exact[i])
            if err > err_max:
                err_max = err
    return err_max


def test_sur_x_carre():
    # deriver f(x) = x^2 et comparer a 2x et 2
    x = linspace(1.0, 3.0, 41)
    dx = x[1] - x[0]

    f = carre(x)
    df_num = derivee_premiere(f, dx)
    d2f_num = derivee_seconde(f, dx)

    df_exact = deux_x(x)
    d2f_exact = deux_constant(x)

    print("=== Test sur f(x) = x^2 (dx =", dx, ") ===")
    print("  derivee premiere : erreur relative max (tout le domaine) =",
          erreur_relative_max(df_num, df_exact))
    print("  derivee premiere : erreur relative max (interieur seul)  =",
          erreur_relative_max(df_num[1:-1], df_exact[1:-1]))
    print("  derivee seconde  : erreur relative max =",
          erreur_relative_max(d2f_num, d2f_exact))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7))
    ax1.plot(x, df_exact, "k-", label="exact : 2x")
    ax1.plot(x, df_num, "ro", markersize=3, label="numerique")
    ax1.set_title("Derivee premiere de x^2")
    ax1.set_xlabel("x"); ax1.set_ylabel("f'(x)"); ax1.legend(); ax1.grid(True)

    ax2.plot(x, d2f_exact, "k-", label="exact : 2")
    ax2.plot(x, d2f_num, "bo", markersize=3, label="numerique")
    ax2.set_title("Derivee seconde de x^2")
    ax2.set_xlabel("x"); ax2.set_ylabel("f''(x)"); ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    plt.show()


def test_convergence_sinus():
    # f(x) = sin(x) : l'erreur centree decroit en O(dx^2)
    for nb_points in [21, 41, 81, 161]:
        x = linspace(0.0, pi, nb_points)
        dx = x[1] - x[0]
        f = sin(x)
        df_num = derivee_premiere(f, dx)
        df_exact = cos(x)
        err = erreur_relative_max(df_num[1:-1], df_exact[1:-1])
        print("  nb_points =", nb_points, " dx =", round(dx, 5),
              " erreur relative interieure =", err)


def main():
    choix = "tout"

    if choix == "x2":
        test_sur_x_carre()
    elif choix == "sinus":
        test_convergence_sinus()
    elif choix == "tout":
        test_sur_x_carre()
        test_convergence_sinus()


if __name__ == "__main__":
    main()
