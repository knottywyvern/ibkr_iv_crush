from ib_insync import *
import sys

def connect_to_tws(host='127.0.0.1', port=7496, clientId=1):
    ib = IB()
    try:
        ib.connect(host, port, clientId)
        print(f"Successfully connected to TWS at {host}:{port}")
        print(f"Server version: {ib.client.serverVersion()}")
        return ib
    except Exception as e:
        print(f"Failed to connect to TWS: {e}")
        sys.exit(1)

def disconnect_from_tws(ib):
    if ib.isConnected():
        ib.disconnect()
        print("Disconnected from TWS")

if __name__ == "__main__":
    util.patchAsyncio()
    
    # Connect to TWS
    ib = connect_to_tws()
    
    # If we get here, connection was successful
    print("Connectivity test successful")
    
    # Disconnect
    disconnect_from_tws(ib)
