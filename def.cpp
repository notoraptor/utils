// Commentaire uni-ligne.
1	byte	unsigned byte	bool	char
2	short	unsigned short
4	int		unsigned int	float
8	long	unsigned long	double

// Booléen
	false // 0
	true // tout nombre non nul
// Entier naturel ou positif
	2; +2;
// Entier négatif
	-2;
// Flottant
	entier (naturel ou négatif)
	2.5; // +2.5
	-2.5;
	3.4e1; // +...
	3.4e-1; // + ...
	-3.4e1;
	-3.4e-1;
// Séquence de bits
	0777; // octal (base 8)
	0xff; // hexadécimal (base 16, 0123456789ABCDEF, A à F peuvent être en minuscule)
	0b011; // binaire (base 2)
// Caractère ASCII étendu (code respecté, vérifier le type de support: `byte` ou `unsigned byte`)
	'c' // Un caractère est un entier relatif

/* Conversions implicites autorisées entre types primitifs. */
// bool <=> tout type
	bool -> ... // false -> 0, true -> 1
	... -> bool // 0 -> false, != 0 -> true
// Entre types de même catégorie, vers le plus grand
	char == byte -> short -> int -> long
	unsigned byte -> unsigned short -> unsigned int  -> unsigned long
	float -> double

/* Toute autre conversion doit être explicit. */
// Avec troncage silencieux:: (type)valeur
	unsigned int ui = 12;
	float f = 1.5;
	long l = (long)ui;	// l == 12
	int i = (int)f;		// i == 1
// Sans troncage (exception à l'exécution si la conversion est impossible): type(valeur)
	double g = 2.0;
	i = int(f); // Erreur à l'exécution: 1.5 n est pas un entier.
	i = int(g); // OK: 2.0 peut devenir un entier.
/** Attention: contrairement au C++, type(valeur) et (type)valeur n'ont pas le même sens ! **/
// Sauf pour bool: bool(a) et (bool)a sont équivalents.

#function type maxof(E, F) {
	return sizeof(E) > sizeof(F) ? E : F;
}

// Opérations.
// a et b doivent être dans la même catégorie (entier signé, entier non signé, flottant)
// Le type bool est dans toutes les catégories.
// Dans une expression mathématique d'entiers signed, bool est automatiquement converti en int.
// Dans une expression mathématique d'entiers unsigned, bool est automatiquement converti en unsigned int.
// Dans une expression mathématique de flottants, bool est automatiquement converti en float.
	double f = true + 2.5f - 4.5d; // == -1
// Dans une expression mathématique de booléens, bool est automatiquement converti en int.
	int a = true + false + true + true; // == 3
E a; F b;
using T = maxof(E, F);
bool not a;
bool a and b;
bool a or b;
bool a xor b;
T a + b;
T a - b;
T a * b;
T a / b;
	Si E et F sont des types entiers, quotient de la division euclidienne.
	Sinon, division complète
T a % b;
	Uniquement pour types entiers. Reste de la division euclidienne.
E a ^ b;
	b doit être un entier
	Retourne un E (type de a)
	si b est signed, a doit être un flottant.

// Opérations bit-à-bit. a et b doivent avoir le même type (a == b).
E bnot a;
E a band b;
E a bor b;
E a bxor b;
// TODO: opérations de décalage:  shift(a, direction, length, boolean_filler=false)

// bloc: {instructions}; // Le ; final est facultatif.
// code: bloc ou instruction.
condition ? instruction;
condition ? {instructions};
condition ? {
	instructions
}
condition ? instruction else instruction;
condition ? {instructions} else {instructions};
condition1 ? instruction1;
else condition2 ? instruction2;
else condition3 ? instruction3;
// ....
else instructionN;


// Chaîne de caractères: ASCII étendu, 0 terminale
	"chaîne"
	char*
// Chaîne de caractères: Unicode (caractères unicode directement dans la chaîne), 0 terminale
	u"avec des caractères unicode dedans"
	unic*
// Chaîne de caractères: HTML
	h"&amp; &#63; chaîn&eacute;e"	// convertie en chaîne unicode, les caractères HTML sont remplacés par les valeurs unicodes correspondants
	unic*