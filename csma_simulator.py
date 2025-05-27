import matplotlib.pyplot as plt
import random
import numpy
import operator

(P, header_bits) = (1800, 240)
(sigma, SIFS, DIFS, ACK_time, tau) = (15, 15, 60, 15, 2)
R = 24E6
T = ((P + header_bits)/R)*1E6
r_max = 3
N = 12

class Station(object):
    def __init__(self, index):
        self.index = index
        self.buffer = Buffer()
        self.state = 'listening'
        self.backoff_time = 0
        self.remaining_time_to_transmit = 0
        self.waiting_time = 0
        self.i = 0
        self.time_waited_for_acknowledgement = 0
        self.acknowledgement_received = False
        self.retransmission_count = 0

    def wait_for_transmission(self):
        self.waiting_time = DIFS
        self.state = 'in_DIFS'

    def pick_backoff_window(self):
        self.backoff_time = random.randint(1, numpy.exp2(self.i-1)*4)
        self.state = 'in_backoff'
        self.i = self.i + 1

    def start_transmission(self):
        self.state = 'transmitting'
        self.remaining_time_to_transmit = ((P + header_bits)/R)*1E6 + tau

    def reset(self):
        self.buffer.removepacket()
        self.state = 'listening'
        self.backoff_time = 0
        self.remaining_bits_to_transmit = 0
        self.waiting_time = 0
        self.i = 0
        self.time_waited_for_acknowledgement = 0
        self.acknowledgement_received = False
        self.retransmission_count = 0

    def enter_backlogged(self):
        self.state = 'backlogged'
        self.backoff_time = 0
        self.waiting_time = 0
        self.remaining_bits_to_transmit = 0
        self.time_waited_for_acknowledgement = 0
        self.acknowledgement_received = False
        self.retransmission_count = 0

    def getbuffer(self):
        return self.buffer

class Buffer(object):
    def __init__(self):
        self.capacity = 1
        self.empty = True
        self.packet_delayed = 0

    def storepacket(self):
        self.empty = False

    def removepacket(self):
        self.empty = True
        self.packet_delayed = 0

class AccessPoint(object):
    def __init__(self, N):
        self.status_table = [0 for i in range(N)]
        self.state = 'not_sending_acknowledgement'
        self.successful_station_index = 0

    def update_status_table(self, transmitting_station_list):
        if len(transmitting_station_list) == 1:
            self.status_table[transmitting_station_list[0]] = 1
        elif len(transmitting_station_list) > 1:
            for index in transmitting_station_list:
                self.status_table[index] = 2

    def request_acknowledgement(self, station_index):
        if self.status_table[station_index] == 1:
            self.state = 'sending_acknowledgement'
            self.successful_station_index = station_index

    def reset(self, station_index):
        self.state = 'not_sending_acknowledgement'
        self.successful_station_index = 0
        self.status_table[station_index] = 0

class MiniSlot(object):
    def __init__(self, start):
        self.duration = sigma
        self.start = start
        self.end = start + self.duration
        self.transmitting_stations = []
        self.state = 'idle'

    def add_to_transmitting_stations_list(self, station_index):
        self.transmitting_stations.append(station_index)

class Simulator(object):
    def __init__(self, N):
        self.N = N
        self.G = numpy.linspace(0.01, 11, 1000)
        self.S_analytical = []
        self.S = []
        self.D_analytical = []
        self.D = []
        self.Expected_N_T_analytical = []
        self.Expected_N_T = []
        self.q_list = numpy.divide(self.G, self.N)

    def compute(self, q):
        stationlist = [Station(index) for index in range(self.N)]
        timeslots = [MiniSlot(i * sigma) for i in range(0, 1001, 1)]
        ap = AccessPoint(self.N)
        p = 1-q
        successfully_transmitted_packets = 0
        packet_delays = []

        for i in range(len(timeslots)):
            if ap.state == 'sending_acknowledgement':
                ssi = ap.successful_station_index
                if stationlist[ssi].time_waited_for_acknowledgement >= (SIFS + tau):
                    stationlist[ssi].acknowledgement_received = True
                    ap.reset(ssi)

            for station in stationlist:
                if not station.buffer.empty:
                    if timeslots[i].state == 'idle' and station.state == 'listening':
                        if random.random() <= p:
                            station.wait_for_transmission()
                    elif timeslots[i].state == 'idle' and (station.state == 'backlogged' or station.state == 'waiting_for_acknowledgement'):
                        if station.retransmission_count > r_max:
                            station.reset()
                        else:
                            if random.random() <= p:
                                station.wait_for_transmission()
                                station.retransmission_count = station.retransmission_count + 1
                    if station.state == 'in_DIFS':
                        if station.acknowledgement_received:
                            successfully_transmitted_packets = successfully_transmitted_packets + 1
                            packet_delays.append(station.buffer.packet_delayed)
                            station.reset()
                        else:
                            if station.waiting_time > 0:
                                station.waiting_time = station.waiting_time - sigma
                                if station.waiting_time == 0:
                                    station.pick_backoff_window()

                            if station.retransmission_count > r_max:
                                station.reset()

                    if station.state == 'in_backoff':
                        if station.backoff_time > 0:
                            station.backoff_time = station.backoff_time - 1
                        elif station.backoff_time <= 0:
                            station.start_transmission()
                            timeslots[i].add_to_transmitting_stations_list(station.index)
                            timeslots[i].state = 'busy'
                    if station.state == 'waiting_for_acknowledgement':
                        if station.acknowledgement_received:
                            successfully_transmitted_packets = successfully_transmitted_packets + 1
                            packet_delays.append(station.buffer.packet_delayed)
                            station.reset()
                        elif not station.acknowledgement_received:
                            if station.time_waited_for_acknowledgement < (SIFS + ACK_time + tau):
                                station.time_waited_for_acknowledgement = station.time_waited_for_acknowledgement + sigma
                            else:
                                station.enter_backlogged()
                    if station.state == 'transmitting':
                        timeslots[i].state = 'busy'
                        station.remaining_time_to_transmit = station.remaining_time_to_transmit - sigma
                        if station.remaining_time_to_transmit <= 0:
                            ap.request_acknowledgement(station.index)
                            station.state = 'waiting_for_acknowledgement'
                            timeslots[i].state = 'idle'
                    station.buffer.packet_delayed = station.buffer.packet_delayed + sigma

                if station.buffer.empty:
                    if random.random() <= q:
                        station.buffer.storepacket()

            ap.update_status_table(timeslots[i].transmitting_stations)

        self.S.append(successfully_transmitted_packets/(1000/(T/sigma)))
        self.D.append(numpy.mean(packet_delays))

    def simulate(self):
        print("Starting simulation...")
        for q in self.q_list:
            print(q)
            self.compute(q)
        self.Expected_N_T = numpy.divide(self.G, self.S)

    def compute_analytical_model(self):
        # Compute throughput and expected number of transmission to success
        for q in self.q_list:
            a = tau / T
            x = (1 - a * q) ** (self.N - 1)
            self.S_analytical.append((a * self.N * q * x) / (1 + a - x))
        self.Expected_N_T_analytical = numpy.divide(self.G, self.S_analytical)

        # Compute HOL delay
        n = self.N

        Ts = DIFS + T + tau + SIFS + ACK_time + tau
        Tc = DIFS + T + tau
        for q in self.q_list:
            p = q
            Ptr = 1-(1-p)**n
            Pi = 1-Ptr
            Psuc = (n-1)*((1-p)**(n-2))
            Pc = 1-Psuc - (1-(1-(1-p)**(n-1)))
            Ps = (1-p)**(n-1)

            E_D = 0
            if Ps > 5e-15:
                for r in range(0, r_max+1):
                    D_r = Ts
                    for i in range(1, r+1):
                        Wi = (2**(i-1))*4
                        D_r = D_r + Wi*(sigma*Pi+Ts*Psuc+Tc*Pc) + Tc
                        E_D = E_D + D_r * ((Ps*((1-Ps)**r))/(1-(1-Ps)**(r_max+1)))
                self.D_analytical.append(E_D/r_max)
            else:
                break

    def plot_performance_curves(self, analytical_model, simulation):
        if simulation:
            self.simulate()
            print("Plotting performance curves...")
            plt.plot(self.G, self.S, label='Simulation')
        if analytical_model:
            self.compute_analytical_model()
            print("Plotting performance curves...")
            plt.plot(self.G, self.S_analytical, label='Analytical')
        plt.ylabel('Throughput S')
        plt.xlabel('Load G')
        plt.title('S vs. G Performance Curves')
        plt.legend()
        plt.show()

        if simulation:
            plt.plot(self.G, self.Expected_N_T, label='Simulation')
        if analytical_model:
            plt.plot(self.G, self.Expected_N_T_analytical, label='Analytical')
        plt.ylabel('Average number of transmission to success E(N_T)')
        plt.xlabel('Load G')
        plt.title('E(N_T) vs. G Performance Curve')
        plt.legend()
        plt.show()

        if simulation:
            plt.plot(self.S, self.D, label='Simulation')
        if analytical_model:
            plt.plot(self.S_analytical[0:len(self.D_analytical)], self.D_analytical, label='Analytical')
        plt.ylabel('Packet Delay D')
        plt.xlabel('Throughput S')
        plt.title('D vs. S Performance Curve')
        plt.legend()
        plt.show()

    def calculate_maximum_throughput(self):
        index, S_max = max(enumerate(self.S_analytical), key=operator.itemgetter(1))
        G_star = self.G[index]
        print("S(max) is: " + repr(S_max))
        print("G* is: " + repr(G_star))

if __name__ == "__main__":
    sim1 = Simulator(N)

    # Compute and plot analytical models
    sim1.plot_performance_curves(analytical_model=True, simulation=False)
    sim1.calculate_maximum_throughput()

    # Simulate and plot the performance curves
    sim1.plot_performance_curves(analytical_model=True, simulation=True)