import flask as fl
import api


app = fl.Flask(__name__)


@app.route('/device_identification/get_vendor/<oui>/<port_list>')
def get_vendor(oui, port_list):
    """
    Parameters:

    oui: First 6 characters of a MAC address.

    port_list: A `.` separated list of ports. For example, if the list of ports
    is 23, 80, and 443, then `port_list` would be this string: "23.80.443". If
    the list of ports is empty, the `port_list` would be this string: "0".

    Examples of valid HTTP requests include:

        `/device_identification/get_vendor/aabbcc/23.80.443`
        `/device_identification/get_vendor/aabbcc/0`

    Returns the name of the vendor or "unknown".

    """
    # Standardize OUI
    oui = oui.lower().replace(' ', '').replace(':', '')[0:6]
    
    # Parse port_list
    if port_list == '.':
        port_list == []
    else:
        port_list = list(set([int(port) for port in port_list.split('.')]))

    return api.get_vendor(oui, type_='port', data=port_list)