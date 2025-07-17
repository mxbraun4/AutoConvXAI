"""Count the number of elements in the data."""


def count_data_points(conversation, parse_text, i, **kwargs):
    """Gets the number of elements in the data.

    Arguments:
        conversation: The conversation object
        parse_text: The parse text for the question
        i: Index in the parse
        **kwargs: additional kwargs
    """
    data = conversation.temp_dataset.contents['X']
    num_elements = len(data)
    ids = list(data.index)
    
    # Store followup for potential follow-up questions
    conversation.store_followup_desc(str(ids))
    
    return {
        'type': 'count',
        'count': num_elements,
        'ids': ids,
        'data_type': 'instances'
    }, 1
