



            # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    # (1) Only mutate if fitness is found to be lower than 5.
    # (2) Randomly choose the number of metrics to be modified, up to half.
    # (3) Randomly choose which such metrics will be modified.
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        if self.fitness() < 5:
            empty_count = 0
            X_count = 0
            loot_count = 0
            M_count = 0
            B_count = 0
            o_count = 0
            pipe_count = 0
            T_count = 0
            E_count = 0

            left = 0
            right = width
            for y in range(height):
                for x in range(left, right):
                    if genome[y][x] == "-":
                        empty_count += 1
                    elif genome[y][x] == "X":
                        X_count += 1
                    elif genome[y][x] == "?":
                        loot_count += 1
                    elif genome[y][x] == "M":
                        M_count += 1
                    elif genome[y][x] == "B":
                        B_count += 1
                    elif genome[y][x] == "o":
                        o_count += 1
                    elif genome[y][x] == "|":
                        pipe_count += 1
                    elif genome[y][x] == "T":
                        T_count += 1
                    elif genome[y][x] == "E":
                        E_count += 1

                    if 
            print('\n')
            print(empty_count)
            print(X_count)
            print(loot_count)
            print(M_count)
            print(B_count)
            print(o_count)
            print(pipe_count)
            print(T_count)
            print(E_count)
            print('\n')            

            return genome