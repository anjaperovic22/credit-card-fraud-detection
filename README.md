# Primena neuronskih mreža za predvidjanje prevara u finansijama (credit card fraud)

## 1. Uvod i opis problema

<p align="justify">
U poslednjih nekoliko godina, sa napretkom tehnologije, većina ljudi koristi kreditne kartice za kupovinu i podmirivanje osnovnih životnih potreba, usled čega i broj prevara povezanih sa njima postepeno raste. Danas gotovo sva preduzeća, od malih firmi do velikih korporacija, koriste kreditnu karticu kao način plaćanja. Prevare sa kreditnim karticama dešavaju se u svim organizacijama, uključujući industriju kućnih aparata, automobilsku industriju, banke i slično. Mnogi procesi, poput rudarenja podataka i algoritamskih pristupa mašinskog učenja, primenjivani su za identifikaciju prevara u transakcijama kreditnih kartica, ali nisu dali značajne rezultate. Zbog toga postoji potreba za razvojem efektivnih i efikasnih algoritama koji rade znatno uspešnije. Mi pokušavamo da predvidimo prevarante u korišćenju kreditne kartice pre nego što transakcija uopšte bude odobrena, i to pomoću algoritma veštačkih neuronskih mreža.
</p>

<p align="justify">
Detekcija prevara sa kreditnim karticama odvija se na sledeći način: korisnik unosi neophodne podatke kako bi izvršio transakciju, a transakcija bi trebalo da bude odobrena tek nakon što se proveri da li u njoj ima bilo kakvih nedozvoljenih aktivnosti. Da bi se to postiglo, detalje transakcije prvo prosleđujemo modulu za verifikaciju, gde se ona klasifikuje u kategoriju prevare ili legitimne transakcije. Svaka transakcija koja bude svrstana u kategoriju prevare se odbija. U suprotnom, transakcija se odobrava.
</p>

<p align="justify">
Glavni tehnički izazov u domenu detekcije finansijskih prevara leži u činjenici da su baze podataka ekstremno debalansirane (engl. <i>class imbalance</i>). U realnim uslovima, regularne transakcije čine preko 99% ukupnog saobraćaja, dok se prevare javljaju u svega nekoliko promila ili procenata. Tradicionalni algoritmi mašinskog učenja često podbace u ovakvim scenarijima jer teže da minimizuju ukupnu grešku tako što će jednostavno svaku transakciju proglasiti legitimnom, čime prividno postižu visoku tačnost. Za razliku od drugih algoritama neuronske mreže, zahvaljujući složenim unutrašnjim slojevima i nelinearnim aktivacionim funkcijama, poseduju jedinstvenu sposobnost da uoče suptilne, skrivene obrasce i anomalije koje opisuju prevaru, čak i u uslovima visoke nesrazmere podataka.
</p>

<p align="justify">
Predmet ovog projektnog zadatka jeste implementacija više modela neuronskih mreža sa ciljem binarne klasifikacije transakcija na legitimne i prevarantske. Pored same konstrukcije i obučavanja mreže, poseban fokus biće stavljen na preprocesiranje podataka, rešavanje problema pomenutog debalansa klasa, kao i na hiperparametarsku optimizaciju kroz promenu stopa učenja, aktivacionih funkcija i optimizacionih algoritama. Evaluacija uspešnosti modela biće izvršena korišćenjem naprednih metričkih pokazatelja.
</p>

## 2. Struktura projekta

<p align="justify">
Kod linearnih algoritama s kojima smo se ranije susretali, poput logističke regresije ili  SVM-a, pretpostavka je linearna. Na primer, mogli bismo reći da što je veći iznos transakcije, veća je šansa da je u pitanju prevara.
</p>
<p align="justify">
U realnosti, prevare su mnogo suptilnije. Izvršena transakcija ne mora da bude u iznosu od više stotina eura da bi bila krađa. Možda vršioci ilegalnih aktivnosti naprave desetak brzih transakcija od po 5 eura u roku od svega par minuta. Dakle, anomalija nije u visokoj vrednosti vež u specifičnoj kombinaciji određenih vrednosti pod čudnim uglovima. 
</p>

<p align="justify">
Glavni cilj ovog rada jeste komparativna analiza performansi različitih arhitektura i konfiguracija veštačkih neuronskih mreža nad prikupljenim podacima, i sve to zarad identifikacije optimalnog modela za primenu u realnom bankarskom poslovanju. 
</p>
<p align="justify">
Detekcija finansijskih prevara zahteva balansiranje između dva dijametralno suprotstavljena poslovna cilja: maksimizacije bezbednosti (hvatanja što većeg broja prevara) i očuvanja vrhunskog korisničkog iskustva (izbegavanje lažnih uzbuna i neopravdanog blokiranja kartica).
</p>
<p align="justify">
Kroz razvoj četiri modela, analiziraćemo kako različite inženjerske metode utiču na ovaj kompromis. 
</p>

* Model 1: **Baseline model (skicit-learn MLPClassifier)**
* Model 2: **Osnovni PyTorch MLP model**
* Model 3: **Težinski PyTorch MLP model**
* Model 4: **Regularizovani PyTorch MLP model**

## 3. Podaci  (izvor, struktura, analiza preprocesiranje)

<p align="justify">
Skup podataka koji će biti korišćen nadalje u radu preuzet je sa <a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud" target="_blank">sledećeg linka</a>.
Dataset sadrži transakcije napravljene kreditnim karticama u septembru 2013. godine od strane evropskih državljana. Sve transakcije obavljene su u periodu od dva dana, gde je utvrđeno postojanje 492 prevare od sveukupno 284.807 izvršenih transakcija. Ovaj skup podataka je kao što smo u uvodu napomenuli - ekstremno debalansiran. 
Što se tiče atributa, sadrži samo numeričke varijable koje većinski predstavljaju rezultat PCA transformacije radi zaštite privatnosti korisnika. Iz tog razloga ne možemo da donosimo ikakve zaključke o pozadini korisnika. 
</p>

<p align="justify">
U opisu skupa podataka na platformi s koje je preuzet, pominje se da su kolone od V1 do V28 rezultat PCA (Principal Component Analysis) transformacije. Najjednostavnije rečeno, PCA je tehnika koja služi za sažimanje podataka, odnosno smanjenje dimenzionalnosti. Najprostiji primer za to šta tačno radi PCA, mogli bismo da uzmemo iz svakodnevnog života. Ukoliko neko želi da fotografiše drvo, on drvo prebacuje iz tri dimenzije u dve dimenzije (fotografiju). Njegov cilj je da zabeleži to drvo sa najviše detalja, senki i oblika, tako da slika bude što verodostojnija i pored toga što je jedna dimenzija nepovratno izgubljena. Ukoliko se odabere loš ugao, neko ko bude posmatrao fotografiju, teže će zaključiti koja je biljka u pitanju. Upravo tako, PCA radi sa podacima – uzima ogroman broj početnih kolona i pronalazi najbolji matematički ugao pod kojim može da ih sabije u manji broj kolona, a da se pritom zadrži suština i ključni šabloni.
</p>

<p align="justify">
U našem skupu podataka, banka je u početku imala vrlo jasne i detaljne podatke o svakoj transakciji: tačnu geografsku lokaciju, ime prodavnice, kategoriju kupljene robe, IP adresu uređaja, istoriju kupovine tog konkretnog klijenta i slično. Pošto su to strogo poverljivi podaci koji narušavaju privatnost korisnika, banka je primenila PCA algoritam. PCA je uzeo sve te stotine tajnih informacija, matematički ih promešao i sabio u 28 anonimnih kolona (od V1 do V28).
</p>

<p align="justify">
Jedine varijable koje su ostale u svom originalnom obliku su <b>Amount</b> i <b>Time</b>. Amount predstavlja iznos transakcije i njena vrednost znatno varira za svaku instancu (može da iznosi od par do više stotina eura). Time sadrži sekunde protekle između svake transakcije i prve transakcije u skupu. Varijabla <b>Class</b> predstavlja ciljnu promenjivu i sadrži vrednosti 1 (za prevaru) i 0 (u suprotnom).
</p>

<p align="justify">
Radi bolje preglednosti, u tabeli ispod je prikazana konačna struktura atributa skupa podataka koji će biti korišćeni za obučavanje neuronske mreže:
</p>

<div align="center">
  
| Naziv atributa | Tip podatka | Opis atributa |
| :--- | :--- | :--- |
| **V1 - V28** | Numerički (Float) | Kompresovane anonimne karakteristike dobijene PCA transformacijom |
| **Time** | Numerički (Float) | Protekle sekunde od prve transakcije u bazi podataka |
| **Amount** | Numerički (Float) | Novčani iznos transakcije (izražen u eurima) |
| **Class** | Binarni (0 ili 1) | **Ciljna promenljiva** (0 = Regularna transakcija, 1 = Prevara) |

</div>

## 4. Arhitektura modela

<div align="center">
<img width="800" height="340" alt="arhitektura_neuronske_mreze" src="https://github.com/user-attachments/assets/0d6c8399-e40f-4920-a3fd-b401796ceafc" />
  <p align="justify">
    <br>
    <b>Slika 1:</b> Detaljan šematski prikaz arhitekture implementiranog višeslojnog perceptrona (MLP). Na šemi je prikazan skraćeni vizuelni pregled slojeva sa tačnim brojem neurona (30 -> 64 -> 32 -> 1).
  </p>
</div>
