# Nyelv felismerés szöveg alapján CNN alkalmazásával

A feladat lényege egy adott szövegről meghatározni, milyen nyelven íródott. 17 nyelv felismerésére képes a program:

- Angol
- Malajálam
- Hindi
- Tamil
- Kannada
- Francia
- Spanyol
- Portugál
- Olasz
- Orosz
- Swéd
- Dán
- Arab
- Török
- Német
- Holland
- Görög

Összesen több mint 10 000 adat található az adatbázisban, ami a Kaggle oldalán található. Ezek az adatok hosszú szövegek, amik alapján betanítható a neurális háló.

Az adathalmaz csv-fájl formátumban letölthető és kettő oszlopot tartalmaz, Text, ami a szövegek oszlopa és Language, ami pedig a szöveg nyelvét jelzi.
## Módosítás az adathalmazban
Nem az eredeti adathalmaz lett felhasználva, hanem egy módosított változata, amiben a legkevesebb adattal rendelkező nyelv ki lett bővítve, azaz több mondatot kapott, hogy ki legyenek jobban egyenlítve a statisztikák. A meglévő Hindi szövegeket felosztásra kerültek, így segítve a tanulást és a Hindi szövegek felismerését.

## Megvalósítás
A feladat Python nyelven készült el, a sok egyedi könyvtár és csomag elérhetőség miatt. A torch Python csomag használatával GPU-n tud futni a modell és annak tanítása, ezáltal is felgyorsítva jelentősen a folyamatot.

Az adatok betöltése után ezeken a szövegeken végigmegy a program, és kitölti a nem egyforma hosszúságú mondatokat üres résszel (padding), így a leghosszabb mondathoz van igazítva az összes többi. Ezeket nyilván tenzorokká kell alakítani, hogy bele lehessen tölteni a modellbe.

A modell elég egyszerű, mivel csak egy dimenzióban dolgozik a program, így egyetlen konvolúciós réteg lett alkalmazva, egy embedding réteggel és MaxPooling réteggel együtt. 

A modell tanítás közben elmentésre kerül, ha az előzőnél jobb százalékot ér el pontosságban, és ez a mentés visszatöltésre kerül a következő indításnál, így nem kell megint megvárni a folyamatot. 


Összesen 50 epoch lenne a tanítási fázis, de a 41. epoch-nál volt a legnagyobb százalék, így az lett elmentve.

Tanítás után a különböző tesztek pontosságát is látni lehet, mint például a Precision, F1 vagy Recall tesztek:


Elindítás előtt a program letölti a különböző csomagokat, amik kellenek a futtatáshoz, majd megnyitja a GUI-t, ahol be lehet illeszteni bármilyen szöveget (képek beillesztésére is van lehetőség, de a nehezebb nyelveket (arab, hindi stb.) nem tudja rendesen kiolvasni).

A szavakat külön-külön nehezen tudja beazonosítani, mivel több nyelvben is lehetnek hasonlóak, vagy akár ugyan olyan szavak mint a vizsgált, de tényleges, hosszabb szövegeknél szinte mindig helyesen meg tudja adni a hozzá tartozó nyelvet.
