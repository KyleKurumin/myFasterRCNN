import pprint

from lxml import etree

file = '2007_000392.xml'

with open(file, 'r') as fd:
    xml_string = fd.read()

obj = etree.fromstring(xml_string)


def parse_xml(xml_obj):
    if len(xml_obj) == 0:
        return xml_obj.text

    result = {}
    for child in xml_obj:
        if child.tag != 'object':
            result[child.tag] = parse_xml(child)
        else:
            result.setdefault('objects', []).append(parse_xml(child))

    return result


pprint.pprint(parse_xml(obj))
