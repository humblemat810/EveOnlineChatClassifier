SELECT mtc.topic_id, ms.message_id,ms.content, tp.name, tp.definition FROM messages AS ms 
RIGHT JOIN message_topic_classifications AS mtc ON ms.message_id=mtc.message_id
LEFT JOIN (
	SELECT id, name, definition FROM topic
	) AS tp
ON mtc.topic_id = tp.id
WHERE ms.message_id IS NOT NULL AND LENGTH(content) >= 1 AND topic_id='crashes'
LIMIT 200