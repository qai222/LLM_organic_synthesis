if (!window.dash_clientside) {
    window.dash_clientside = {}
}

window.dash_clientside.clientside = {
    update_reaction: function(reaction_id, clean_data) {
	    return clean_data[reaction_id]
        }
    }