Nume: Coitu Sebastian-Teodor  
Grupa: 334CA

# Tema 2

## Organizare

### Explicatie pentru solutia aleasa

**1. Calculul arborelui Merkle (**`construct_merkle_root`**):**

Am abordat reducerea arborelui pe niveluri prin folosirea un kernel pentru paralelizarea initiala a frunzelor (1 thread = 1 tranzactie), urmat de un kernel de reducere (pentru fiecare nou parinte se injumatateste numarul de hash-uri) . Pentru a evita overhead-ul alocarilor dinamice la fiecare apel, am folosit lazy allocation si o tehnica de "ping-pong" a pointerilor intre doua buffere pre-alocate in memoria globala. Primul buffer retine hash-urile de la nivelul curent, iar al doilea este populat cu hash-uri calculate pentru nivelul urmator. La fiecare iteratie a arborelui, pur si simplu interschimb pointerii celor doua buffere.

**2. Gasirea Nonce-ului (`find_nonce`):**

Aici a avut ceva mai mult de treaba pe partea de optimizare. Am pornit de la o implementare initiala destul de straight-forward, de tipul 1 thread = 1 nonce. Acest lucru a necesitat calcularea unui numar imens de blocuri pe baza valorii max_nonce (ducand si la un integer overflow si practic ruland kernelul cu configuratia <<<0, 256>>>). Totusi, testarea a demonstrat ca aruncarea a peste 16 milioane de blocuri catre GPU creeaza un overhead masiv pe scheduler, obtinand un timp de putin peste 0.04s.

Astfel, am trecut la o implementare si mai optimizata: Grid-Stride Loop. Am lansat un grid fix, dimensionat pentru a satura complet cele 108 SM-uri ale GPU-ului (1024 blocuri x 256 thread-uri inseamna ca fiecare SM are 9-10 blocuri de procesat). Acest load suplimentar ascunde latenta generata de citiri din VRAM, deoarece cand un set de thread-uri se blocheaza, SM-ul pune acel set pe pauza si trece  la alt set de thread-uri care au datele pregatite. Acestea recicleaza resursele si itereaza peste spatiul de cautare folosind un stride egal cu dimensiunea grilei:

```cpp
uint64_t stride = (uint64_t)blockDim.x * gridDim.x;
for (uint64_t nonce = idx; nonce < max_nonce; nonce += stride) { ... }
```

Aceasta abordare a scazut timpul de executie la ~0.006s. Pentru oprirea imediata a executiei odata ce o solutie a fost gasita, am folosit un flag global marcat ca `volatile` (pentru a forta citirea direct din VRAM, evitand cache-ul registrelor) si `atomicMin` pentru evitarea unui race condition la scrierea nonce-ului castigator. Hash-ul final al blocului este recalculat pe CPU pentru a elimina operatii complexe de copiere din GPU.

---

**Consideri ca tema este utila?**

Da, consider ca este foarte utila, deoarece ofera o perspectiva practica asupra modului in care anumite limitari hardware (cum ar fi memoria stivei, planificarea blocurilor) influenteaza arhitectura software. Acest exercitiu m-a ajutat sa fac o tranzitie de la gandirea secventiala la modelul SIMT.

**Consideri implementarea naiva, eficienta, se putea mai bine?**

Implementarea finala bazata pe Grid-Stride Loop este foarte eficienta si foarte flexibila pe diferite arhitecturi. Spre deosebire de o implementare naiva (care nu a livrat rezultatele asteptate) dependenta strict de datele de intrare, solutia curenta scaleaza automat, fiind capabila sa ruleze la eficienta maxima pe orice placa video, indiferent de numarul de nuclee disponibile.

**Cazuri speciale:**

- **Nonce-uri multiple valide:** Intr-o rulare paralela masiva, GPU-ul gaseste primul nonce valid din punct de vedere temporal, care nu coincide mereu cu minimul absolut gasit de CPU (care cauta strict secvential de la 0 la N). Desi rezultatele obtinute cu `diff` difera pe coloana nonce-ului fata de referinta CPU, solutia obtinuta de GPU este, de asemenea corecta.

---

## Implementare

- Intregul enunt al temei este implementat, atat `construct_merkle_root` cat si `find_nonce` fiind paralelizate si functionale.
- **Functionalitati extra:** Am integrat un macro `CHECK_CUDA` pentru a prinde erorile care apar silentios pe fundal, functionalitate care s-a dovedit vitala pe parcursul dezvoltarii.

**Dificultati intampinate:**

- **Erori silentioase de GPU si mediul de rulare:** Kernel-urile esuau silentios la un moment dat, returnand un Merkle Root gol. Am integrat un macro `CHECK_CUDA` pentru a prinde aceste erori si alaturi de utilitarul `nvidia-smi`, am descoperit ca nu aveam un GPU disponibil in sesiunea curenta. Rularea recomandata in enunt (cea prin Makefile) a rezolvat problema (ramasesem cu ideea de la prima tema, unde erau destule noduri disponibile in care puteam sta fara sa deranjez pe altii).
- **Depasirea limitelor de date:** Crearea initiala a grilei pentru limitele lui `UINT32_MAX` genera Integer Overflow din cauza operarii pe 32 de biti a variabilelor de block-size.

**Lucruri interesante descoperite pe parcurs:**

- Am putut observa fizic pe timpii de rulare diferenta dintre a trimite 16 milioane de blocuri usor incarcate catre driver-ul NVIDIA vs 1024 de blocuri foarte incarcate (metoda Grid-Stride pe care am descoperit-o).

---

## Resurse utilizate

- Laboratoarele de ASC
- Documentatia oficiala NVIDIA: [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/).
- Acest articol despre procesarea datelor in CUDA: [https://victorleungtw.medium.com/accelerating-data-processing-with-grid-stride-loops-in-cuda-8adc810d188d](https://victorleungtw.medium.com/accelerating-data-processing-with-grid-stride-loops-in-cuda-8adc810d188d)
- Ajutor de la LLM

