Az eredeti programot és az abból megkapott eredményeket felhasználva a programot kiegészítettem a következő funkciókkal:

- Az elemzett szöveg szavai, a térben hozzájuk rendelt vektorok és azok 2 dimenzióra való leképezésével kapott koordináták felhasználásával az iGraph lib által megvalósított rajzoló metódusokkal ábrázolja a gráfot.

- Az eredeti funkcionalitásnak megfelelően a program által egymáshoz közelinek ítélt szavak a koordináta-rendszerben is egymáshoz közeli pontokban helyezkednek el.

- A korábbi tesztek és megfigyelések alapján az egymástól 2 egységnyire található szavakat ábrázoló pontok azok, amelyek között valóban szoros összefüggés fedezhető fel. Például: szinonimák, azonos szófajúak, a vizsgált szövegben többször előfordulnak egymás környezetében.

- Az egymástól 2 egységnyi sugáron belül elhelyezkedő pontok megállapításához felhasználtam a scipy lib K-dimenziós fáját (ami a szavak koordinátáit tárolja) és a hozzá tartozó, adott egységnyi sugáron (jelen esetben 2 egység) belüli koordináta-párokat kiválasztja és egy listába gyűjti

- A gráfot kirajzoló metódus figyelembe veszi az egymástól adott távolságra lévő pontokat, és csak azok közé húz élt, ahol ez a távolság kisebb vagy egyenlő a megadott 2 egységnél

- A gráf nem tartalmaz hurkokat és többszörös éleket, tehát a végeredmény minden esetben egy egyszerű gráf


A program tesztelésére felhasznált tesztbemenetek (mindegyik .txt formátumban, speciális karakterek nélkül):

Az eredeti tesztprogramhoz mellékelt szövegfájl.

Generált 10.000 szavas lorem ipsum szövegfájl.

Részlet a bibliai Jelenések könyvéből.

Részlet Charles Dickens Twist Olivér c. regényéből.

Részlet a Király Beszéde c. film forgatókönyvéből.

Az összes részlethez generált szóháló a megadott mappákban található a github repository főkönyvtárában.

Mindegyik bemenet az elvárásoknak megfelelő szóhálót eredményezett - hosszabb szövegek esetén az eredmények is pontosabbak lettek, viszont a folyamat sokkal több rendszermemóriát igényelt, a default szöveg esetében többet is annál, mint amennyi a programot futtató számítógépnek rendelkezésére állt. A többi, hosszabb szövegek esetében a szóhálók nagyon sűrűk, de vektor grafikus fájlként tetszés szerint nagyíthatók a további részletekért.

A túlzottan sűrű szóhálók visszavezethetőek arra, hogy a program által feldolgozott szövegek nem elég hosszúak/tartalmaznak elegendő különböző szót, tehát a programnak nem áll rendelkezésére egy nagy méretű tanítóminta, amivel pontosabban meg tudná határozni, mely szavak azok, amik valóban közelebb állnak egymáshoz. A probléma kiküszöbölhető egy megfelelő szöveggel, azonban ennek a vizsgálata több rendszermemóriát is igényel. Valamint mivel ez csak egy nagyon alapfokú implementációja a word2vec-nek és a hozzá kapcsolódó technológiáknak, egy jobban személyre szabható programverzióval többet is el lehetne érni mind teljesítmény, mind erőforrás-hatékonyság terén.


