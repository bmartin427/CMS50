#!/usr/bin/env python3
# Copyright 2019 Brad Martin.  All rights reserved.

"""This script supports serial communication with a Contec CMS50D+ pulse
oximeter at firmware version 4.6, and potentially other similar sensors.
"""

import argparse
import numpy
import numpy.ma.mrecords
import serial
import sys


class ProtocolSyncError(RuntimeError):
    pass


class Cms50Driver:
    # NOTE: Documentation for this protocol was found here:
    # https://github.com/albertcbraun/CMS50FW/blob/master/Communication%20protocol%20of%20pulse%20oximeter%20V7.0.pdf

    # 'Data type' values.  These first values are used in device-to-PC
    # communication:
    _DT_REALTIME_DATA = 0x01
    _DT_DEVICE_IDS = 0x04
    _DT_USER_INFO = 0x05
    _DT_STORAGE_START_DATE = 0x07
    _DT_STORAGE_DATA_LEN = 0x08
    _DT_STORAGE_DATA_PI = 0x09
    _DT_STORAGE_DATA_SEGMENT_AMOUNT = 0x0A
    _DT_CMD_FEEDBACK = 0x0B
    _DT_FREE_FEEDBACK = 0x0C
    _DT_DISCONNECT_NOTICE = 0x0D
    _DT_PI_IDS = 0x0E
    _DT_STORAGE_DATA = 0x0F
    _DT_USER_AMOUNT = 0x10
    _DT_DEVICE_NOTICE = 0x11
    _DT_STORAGE_START_TIME = 0x12
    _DT_STORAGE_DATA_IDS = 0x15
    # These values are used in PC-to-device communication:
    _DT_SET_DEVICE_ID = 0x04
    _DT_CTRL_CMDS = 0x7D

    # 'Command' values.  These are sent with the _DT_CTRL_CMDS data type.
    _CMD_START_REALTIME_DATA = 0xA1
    _CMD_STOP_REALTIME_DATA = 0xA2
    _CMD_QUERY_STORAGE_DATA_SEGMENT_AMOUNT = 0xA3
    _CMD_QUERY_STORAGE_DATA_LEN = 0xA4
    _CMD_QUERY_STORAGE_DATA_START = 0xA5
    _CMD_QUERY_STORAGE_DATA = 0xA6
    _CMD_STOP_STORAGE_DATA = 0xA7
    _CMD_QUERY_DEVICE_IDS = 0xAA
    _CMD_QUERY_USER_INFO = 0xAB
    _CMD_QUERY_PI_SUPPORT = 0xAC
    _CMD_QUERY_USER_AMOUNT = 0xAD
    _CMD_DELETE_STORAGE_DATA = 0xAE
    _CMD_HEARTBEAT = 0xAF
    _CMD_QUERY_DEVICE_NOTICE = 0xB0
    _CMD_SYNC_TIME = 0xB1
    _CMD_SYNC_DATE = 0xB2
    _CMD_QUERY_STORAGE_DATA_IDS = 0xB6

    # Reason codes.  Used when rejecting a command, or signaling disconnect.
    _REASON_COMPLETED = 0x00
    _REASON_SHUTDOWN = 0x01
    _REASON_USER_CHANGE = 0x02
    _REASON_RECORDING = 0x03
    _REASON_STORAGE_DELETE_FAIL = 0x04
    _REASON_UNSUPPORTED = 0x05
    _REASON_UNKNOWN = 0xFF

    def __init__(self, device):
        self._port = serial.Serial(device, baudrate=115200, timeout=2)

    def query_storage_data_length(self, user=0, segment=0):
        self._send_cmd(self._CMD_QUERY_STORAGE_DATA_LEN,
                       bytearray([user, segment]))
        data = self._expect_packet(self._DT_STORAGE_DATA_LEN)
        assert data[0] == user
        assert data[1] == segment
        return data[2] + (data[3] << 8) + (data[4] << 16) + (data[5] << 24)

    def query_storage_data(self, user=0, segment=0,
                           expected_length=None, expect_pi=None):
        if expected_length is None:
            expected_length = self.query_storage_data_length(
                user=user, segment=segment)
        if expect_pi is None:
            # TODO(bmartin) Query whether PI data is present (STORAGE_DATA_IDS).
            expect_pi = False
        self._send_cmd(self._CMD_QUERY_STORAGE_DATA,
                       bytearray([user, segment]))
        data = bytearray()
        while len(data) < expected_length:
            d = self._expect_packet(self._DT_STORAGE_DATA_PI if expect_pi
                                    else self._DT_STORAGE_DATA)
            data += d
            print('Downloaded %d/%d bytes\r' % (len(data), expected_length),
                  end='')
        data = data[:expected_length]
        print()

        dtype = [('spO2_pct', 'u1'), ('pulse_rate', 'u1')]
        if expect_pi:
            dtype += [('pi', '<u2')]
        records = numpy.rec.fromstring(data, dtype=dtype)
        # Prepend a relative timestamp to the data.
        timed_records = numpy.recarray(
            len(records), dtype=[('relative_time', numpy.int32)] + dtype)
        timed_records.relative_time = numpy.arange(len(records))
        for d in dtype:
            timed_records[d[0]] = records[d[0]]
        records = timed_records.view(numpy.ma.mrecords.mrecarray)
        # Exclude special values.
        numpy.ma.masked_equal(records.spO2_pct, 0, copy=False)
        numpy.ma.masked_equal(records.spO2_pct, 0x7F, copy=False)
        numpy.ma.masked_equal(records.pulse_rate, 0, copy=False)
        numpy.ma.masked_equal(records.pulse_rate, 0xFF, copy=False)
        if expect_pi:
            numpy.ma.masked_equal(records.pi, 0, copy=False)
            numpy.ma.masked_equal(records.pi, 0xFFFF, copy=False)
            # Convert units after replacing magic binary values.
            assert dtype[-1][0] == 'pi'
            dtype[-1][1] = numpy.float32
            records = records.astype(dtype)
            records.pi /= 100.
        return records

    def _send_packet(self, data_type, data=None):
        if data is None:
            data = []
        assert len(data) <= 7
        msg = bytearray(len(data) + 2)
        msg[0] = data_type
        msg[1] = 0x80
        for i, b in enumerate(data):
            if b & 0x80:
                msg[1] |= (1 << i)
            msg[i + 2] = b | 0x80
        self._port.write(msg)

    def _get_packet(self):
        data_type = self._get_packet_data_type()
        data_len = self._data_len_for_data_type(data_type)
        data = self._get_packet_data(data_len)
        return data_type, data

    def _get_packet_data_type(self):
        while True:
            b = self._port.read()
            if not len(b):
                raise serial.SerialTimeoutException('read data type timeout')
            assert len(b) == 1
            if not (b[0] & 0x80):
                return b[0]
            raise ProtocolSyncError('Expected data type, got byte %r' % b.hex())

    @staticmethod
    def _data_len_for_data_type(dt):
        C = Cms50Driver
        if dt in [C._DT_REALTIME_DATA, C._DT_DEVICE_IDS, C._DT_USER_INFO,
                  C._DT_DEVICE_NOTICE, C._DT_STORAGE_DATA_IDS, C._DT_CTRL_CMDS]:
            return 7
        if dt in [C._DT_STORAGE_START_DATE, C._DT_STORAGE_DATA_LEN,
                  C._DT_STORAGE_DATA, C._DT_STORAGE_START_TIME]:
            return 6
        if dt in [C._DT_STORAGE_DATA_PI]:
            return 4
        if dt in [C._DT_STORAGE_DATA_SEGMENT_AMOUNT, C._DT_CMD_FEEDBACK]:
            return 2
        if dt in [C._DT_DISCONNECT_NOTICE, C._DT_PI_IDS, C._DT_USER_AMOUNT]:
            return 1
        if dt in [C._DT_FREE_FEEDBACK]:
            return 0
        raise ValueError('Unknown data type %r' % dt)

    def _get_packet_data(self, data_len):
        assert data_len >= 0
        assert data_len <= 7
        count = data_len + 1
        msg = self._port.read(count)
        if not all(b & 0x80 for b in msg):
            raise ProtocolSyncError('Expected data, got bytes %r' % msg.hex())
        if len(msg) < count:
            raise serial.SerialTimeoutException(
                'read data timeout (got %d bytes, wanted %d)' %
                (len(msg), count))
        assert len(msg) == count
        data = bytearray(msg[1:])
        for i in range(data_len):
            if not (msg[0] & (1 << i)):
                data[i] &= 0x7F
        return data

    def _send_cmd(self, cmd, data=None):
        if data is None:
            data = []
        assert len(data) <= 6
        # Packet must be padded to the maximum length.
        msg = bytearray([cmd]) + data + bytearray([0] * (6 - len(data)))
        self._send_packet(self._DT_CTRL_CMDS, msg)

    def _expect_packet(self, dt):
        got_dt, got_data = self._get_packet()
        if got_dt == dt:
            return got_data
        if got_dt == self._DT_CMD_FEEDBACK:
            if got_data[1] == self._REASON_UNSUPPORTED:
                raise NotImplementedError(
                    'Device reports command %r unsupported' % got_data[:1])
            raise RuntimeError(
                'Device reports command %r failed for reason %r' %
                (got_data[:1], got_data[1:2]))
        raise RuntimeError(
            "Got unexpected data type '%02X' when '%02X' expected "
            "(payload: %r)" %
            (got_dt, dt, got_data))


def write_csv(data, filename):
    # Annoyingly, numpy.savetxt() doesn't handle masked data.  If we want blank
    # entries where masked data would be, we'll have to generate the strings
    # directly.
    dtype = [(n, '<U6') for n in data.dtype.names]
    txt = data.astype(dtype)
    numpy.savetxt(filename, txt.filled(fill_value=''), delimiter=',', fmt='%s',
                  header=','.join(data.dtype.names))

def read_csv(filename):
    # The underlying data type for the fields will change to 32-bit from 8- or
    # 16-bit following a write/read cycle.  This is probably harmless.
    return numpy.genfromtxt(
        filename, dtype=numpy.int32, delimiter=',',
        names=True, usemask=True).view(numpy.ma.mrecords.mrecarray)


def plot_data(data):
    import matplotlib.pyplot as plt

    plt.figure('SpO2')
    H = 3600.
    plt.plot(data.relative_time / H, data.spO2_pct, color='b')
    plt.plot([data.relative_time[0] / H, data.relative_time[-1] / H], [88, 88],
             color='r', linestyle='--')
    plt.xlabel('relative time (h)')
    plt.ylabel('SpO2 %')

    plt.figure('SpO2 histogram')
    plt.hist(data.spO2_pct)

    plt.show()


def analyze_data(data, plot):
    _LOW_SPO2_PCT = 88
    print('Minimum spO2: %d%%' % numpy.min(data.spO2_pct))
    print('Time under %d%% spO2: %.1f min' %
          (_LOW_SPO2_PCT, numpy.sum(data.spO2_pct < _LOW_SPO2_PCT) / 60.))
    # TODO(bmartin) 'Events'?
    if plot:
        plot_data(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-d', '--device', help='Serial device to use')
    parser.add_argument(
        '-D', '--download-csv',
        help='Download stored record to CSV with given filename')
    parser.add_argument(
        '-i', '--input-csv',
        help='Skip device communication, and read data from given CSV filename')
    parser.add_argument(
        '-p', '--plot', action='store_true',
        help='Plot downloaded or loaded data')
    args = parser.parse_args()

    if args.input_csv:
        data = read_csv(args.input_csv)
        analyze_data(data, args.plot)
        return

    driver = Cms50Driver(args.device)
    if not args.download_csv:
        raise NotImplementedError('live operation not supported yet')

    data_len = driver.query_storage_data_length()
    if not data_len:
        print('Device reports no stored data!')
        return 1
    print('Device reports %r bytes of stored data' % data_len)
    data = driver.query_storage_data(expected_length=data_len)
    print('Retrieved %r records' % data.shape[0])
    write_csv(data, args.download_csv)
    print('Wrote %r' % args.download_csv)
    analyze_data(data, args.plot)


if __name__ == '__main__':
    sys.exit(main())
